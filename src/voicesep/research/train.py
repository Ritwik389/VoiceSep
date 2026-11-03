from pathlib import Path
from dataclasses import asdict
import json
import hashlib
import random
import time

import numpy as np
import torch
import yaml

from ..models.separator import ModelConfig, VoiceSeparator
from ..losses import hungarian_loss
from .data import MixtureFactory, read_jsonl, assert_disjoint
from .evaluate import score_mixture


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, draw_index, config,
                    best_score=None, validation=None, stale=0):
    checkpoint = {"format_version": 1, "model_config": asdict(model.config), "model": model.state_dict(),
                  "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                  "scaler": scaler.state_dict(), "step": step, "draw_index": draw_index,
                  "training_config": config, "trained": step > 0, "validation": validation or {},
                  "best_score": best_score, "stale_evaluations": stale,
                  "calibration": config.get("calibration", {}),
                  "rng": {"torch": torch.get_rng_state(), "python": random.getstate(),
                          "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}
    manifest = Path(config["data"]["manifest"])
    validation_manifest = manifest.with_name("validation.jsonl")
    checkpoint["data_fingerprints"] = {
        name: hashlib.sha256(candidate.read_bytes()).hexdigest()
        for name, candidate in (("train", manifest), ("validation", validation_manifest),
                                ("validation_mixtures", Path(config["data"].get("validation", "missing"))))
        if candidate.is_file()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(model, source_manifest, recipes_path, device, calibration):
    from ..inference import detect_streams, Calibration
    factory = MixtureFactory(source_manifest, sample_rate=model.config.sample_rate, slots=model.config.slots)
    model.eval()
    groups = {}
    for recipe in read_jsonl(recipes_path):
        mixture, references = factory.render(recipe)
        prediction = model(torch.from_numpy(mixture).unsqueeze(0).to(device))
        waves = prediction["waveforms"][0].cpu().numpy()
        probs = prediction["activity_logits"][0].sigmoid().cpu().numpy()
        accepted, _ = detect_streams(waves, probs, mixture, Calibration(**calibration))
        score = score_mixture(waves[accepted], references, mixture, compute_stoi=False)
        groups.setdefault((recipe["condition"], recipe["speaker_count"]), []).append(
            score["si_sdri"] - 5 * abs(score["predicted_count"] - score["reference_count"]))
    model.train()
    # Each condition/count group is equally weighted, independent of sample imbalance.
    return float(np.mean([np.mean(values) for values in groups.values()]))


def _optimizer_step(model, optimizer, scaler, clip_norm):
    previous_scale = scaler.get_scale()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm,
                                   error_if_nonfinite=not scaler.is_enabled())
    # GradScaler skips the update and lowers its scale when unscale_ detected overflow.
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous_scale


def train(config_path, output_dir, pretrained=None, resume=None, device=None,
          steps_override=None, benchmark=False, segment_seconds=None, max_seconds=None):
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("The wall-time budget must be positive")
    config = yaml.safe_load(Path(config_path).read_text())
    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Select a free notebook GPU or pass --device cpu for a smoke test.")
    c = ModelConfig(**config["model"])
    if c.architecture == "convtasnet":
        from ..models.baseline import ConvTasNetSeparator
        model = ConvTasNetSeparator(c)
    else:
        model = VoiceSeparator(c)
    t, d = config["training"], config["data"]
    if segment_seconds is not None:
        d["segment_seconds"] = segment_seconds
    if not resume:
        if c.architecture == "convtasnet":
            model.initialize_from_official(pretrained)
        elif pretrained:
            model.initialize_from_official(pretrained)
        else:
            raise ValueError("Fine-tuning requires --pretrained; no silent random initialization")
    model.to(device)
    head_ids = {id(p) for p in model.activity.parameters()}
    if c.architecture == "tflocoformer":
        head_ids |= {id(p) for p in model.backbone.deconv.parameters()}
    else:
        head_ids |= {id(p) for p in model.backbone.masker.mask_net.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": t["backbone_lr"]},
        {"params": [p for p in model.parameters() if id(p) in head_ids], "lr": t["head_lr"]}], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    use_amp = t.get("amp", True) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=1024)
    step, draw_index, best_score, stale = 0, 0, None, 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["model_config"] != asdict(c) or state["training_config"] != config:
            raise ValueError("Resume config must match the checkpoint (use --steps to change only the run budget)")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, draw_index, best_score = state["step"], state["draw_index"], state.get("best_score")
        stale = state.get("stale_evaluations", 0)
        torch.set_rng_state(state["rng"]["torch"])
        random.setstate(state["rng"]["python"])
        if device.startswith("cuda") and state["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    data = MixtureFactory(d["manifest"], c.sample_rate, d["segment_seconds"], c.slots, seed,
                          d.get("partial_overlap_probability", 0), d.get("noise_probability", 0),
                          d.get("reverb_probability", 0))
    if resume and state.get("data_fingerprints", {}).get("train") not in (None, data.manifest_sha256):
        raise ValueError("Training source manifest changed since the saved checkpoint")
    validation_source = str(Path(d["manifest"]).with_name("validation.jsonl"))
    validation_path = d.get("validation")
    if not benchmark:
        if not validation_path or not Path(validation_path).is_file():
            raise ValueError("Frozen validation mixtures are required before training; run voicesep freeze")
        assert_disjoint(d["manifest"], validation_source)
        if resume:
            for key, candidate in (("validation", validation_source), ("validation_mixtures", validation_path)):
                expected = state.get("data_fingerprints", {}).get(key)
                if expected is not None and hashlib.sha256(Path(candidate).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"{key} changed since the saved checkpoint")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = step + 100 if benchmark else (steps_override if steps_override is not None else t["steps"])
    model.train()
    started = time.perf_counter()
    losses = []
    completed_draw_index = draw_index
    skipped_updates = consecutive_overflows = 0
    try:
        while (step < limit and stale < t.get("early_stopping_evaluations", 5)
               and (max_seconds is None or time.perf_counter() - started < max_seconds)):
            model.freeze_backbone(False if benchmark else step < t["warmup_steps"])
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for _ in range(t["accumulation"]):
                mixture, references, _ = data.draw(draw_index)
                draw_index += 1
                mixture = torch.from_numpy(mixture).unsqueeze(0).to(device)
                references = torch.from_numpy(references).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=use_amp):
                    output = model(mixture)
                    loss, _ = hungarian_loss(output, references, mixture, c.hop, c.n_fft)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss / t["accumulation"]).backward()
                total += float(loss.detach()) / t["accumulation"]
            updated = _optimizer_step(model, optimizer, scaler, t["clip_grad_norm"])
            completed_draw_index = draw_index
            if not updated:
                skipped_updates += 1
                consecutive_overflows += 1
                print(json.dumps({"amp_overflow_skipped": skipped_updates, "step": step,
                                  "scale": scaler.get_scale(), "draw_index": draw_index}), flush=True)
                if consecutive_overflows >= 16:
                    raise FloatingPointError('Persistent AMP gradient overflow after 16 scale reductions')
                continue
            consecutive_overflows = 0
            step += 1
            losses.append(total)
            record = {"step": step, "loss": total, "draw_index": draw_index,
                      "elapsed_seconds": time.perf_counter() - started}
            with (output_dir / "train.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            if step == 1 or step % 10 == 0:
                print(json.dumps(record), flush=True)
            if not benchmark and step % t["validate_every"] == 0:
                score = validate(model, validation_source, validation_path, device, config.get("calibration", {}))
                scheduler.step(score)
                validation_record = {"step": step, "validation_balanced_score": score,
                                     "learning_rates": [group['lr'] for group in optimizer.param_groups]}
                with (output_dir / 'validation.jsonl').open('a') as log:
                    log.write(json.dumps(validation_record) + '\n')
                print(json.dumps(validation_record), flush=True)
                if best_score is None or score > best_score + 0.05:
                    best_score, stale = score, 0
                    save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, scaler,
                                    step, draw_index, config, best_score, {"balanced_score": score, "release_ready": False}, stale)
                else:
                    stale += 1
            if step % t["checkpoint_every"] == 0:
                save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, step, draw_index, config, best_score, stale=stale)
            if stale >= t.get("early_stopping_evaluations", 5):
                break
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError("GPU memory exhausted. Resume from last.pt or rerun the benchmark with --segment-seconds 2.")
    finally:
        # Saves only completed optimizer steps; an interrupted accumulation is replayed from its step boundary.
        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, scaler, step,
                        completed_draw_index, config, best_score, stale=stale)
    report = {"completed_steps": step, "steps_this_run": len(losses),
              "seconds_per_step": (time.perf_counter() - started) / max(len(losses), 1),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
              "device": device, "benchmark_only": benchmark, "release_ready": False,
              "amp_skipped_updates": skipped_updates,
              "stopped_for_time_budget": max_seconds is not None and step < limit
              and time.perf_counter() - started >= max_seconds}
    (output_dir / "throughput.json").write_text(json.dumps(report, indent=2))
    return report
