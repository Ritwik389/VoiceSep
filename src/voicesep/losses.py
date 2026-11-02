"""Silence-safe Hungarian PIT and matched activity supervision."""
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def si_sdr(estimate, reference, eps=1e-8):
    estimate = estimate - estimate.mean(-1, keepdim=True)
    reference = reference - reference.mean(-1, keepdim=True)
    projection = (estimate * reference).sum(-1, keepdim=True) * reference / (
        reference.square().sum(-1, keepdim=True) + eps)
    value = 10 * torch.log10((projection.square().sum(-1) + eps) /
                             ((estimate - projection).square().sum(-1) + eps))
    return torch.where(estimate.square().sum(-1) <= eps, torch.full_like(value, -30.), value)


def activity_targets(references, frame_count, n_fft=128, hop=64):
    b, s, n = references.shape
    power = F.avg_pool1d(references.reshape(b * s, 1, n).square(), n_fft, hop,
                         padding=n_fft // 2).reshape(b, s, -1)
    power = F.interpolate(power, size=frame_count, mode="nearest")
    threshold = power.amax(-1, keepdim=True) * 1e-4
    return ((power > threshold) & (power > 1e-9)).float()


def hungarian_loss(output, references, mixture, hop=64, n_fft=128):
    predictions = output["waveforms"].float()
    logits = output["activity_logits"].float()
    if predictions.shape != references.shape:
        raise ValueError("Predictions and padded references must have matching shape")
    activity = activity_targets(references, logits.shape[-1], n_fft, hop)
    active = references.square().mean(-1) > 1e-9
    batch_losses, assignments = [], []
    for b in range(len(predictions)):
        p, r = predictions[b], references[b]
        scale = mixture[b].square().mean().clamp_min(1e-6)
        costs = []
        for target in range(r.shape[0]):
            if active[b, target]:
                # Resolve SI-SDR's scale ambiguity so exported tracks retain usable levels.
                wave_cost = -si_sdr(p, r[target].expand_as(p)).clamp(-30, 30)
                wave_cost = wave_cost + 0.1 * (p - r[target]).abs().mean(-1) / scale.sqrt()
            else:
                wave_cost = 10 * torch.log1p(p.square().mean(-1) / scale)
            act_cost = F.binary_cross_entropy_with_logits(
                logits[b], activity[b, target].expand_as(logits[b]), reduction="none").mean(-1)
            costs.append(wave_cost + 0.5 * act_cost)
        cost = torch.stack(costs, dim=1)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        assignments.append((rows.tolist(), cols.tolist()))
        reconstruction = (p.sum(0) - r.sum(0)).abs().mean() / scale.sqrt()
        batch_losses.append(cost[rows, cols].mean() + 0.1 * reconstruction)
    return torch.stack(batch_losses).mean(), assignments
