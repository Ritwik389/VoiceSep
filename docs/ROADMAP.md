# Ordered release gates

1. **Model milestone:** actual free-GPU fine-tuning, calibrated automatic counts, fixed English/Hindi evaluation, representative permitted audio, ≥10 dB SI-SDRi and ≥90% count accuracy in every 4/5-speaker language condition, human listening review.
2. **V1:** tested local editor, CLI, notebooks, API, transcripts/activity/remix/exports, reproducible GitHub source and externally stored checkpoint. Until the model milestone passes, this is an implementation preview, not a working five-speaker release.
3. **V1.1 priority:** the user-prioritized age/vocal-presentation/emotion preview is implemented with local Hugging Face models on the existing 8 kHz tracks. Next evaluate attribute stability and labeled English/Hindi accuracy, then upgrade separation to native 16 kHz. Speaking rate and pitch follow. Respect noncommercial audEERING weight terms. See [the next-phase plan](SPEAKER_ANALYSIS_PLAN.md).
4. **V2:** real recordings, noise/reverb robustness, then broader multilingual validation once English/Hindi pass.
5. **Scaling:** six/eight/ten-speaker recursive/attractor comparisons and quality-versus-count curves.
6. **Live:** a causal/distilled model, buffering, persistent identities, latency and streaming evaluation. Offline chunking is not evidence of live capability.
