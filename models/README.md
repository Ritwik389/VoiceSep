# Checkpoints

Model binaries are ignored by Git. Run `voicesep download-backbone` to obtain and checksum the official two-speaker TF-Locoformer checkpoint (~60 MB). It is not a trained five-speaker separator.

Fine-tune using the free GPU notebook, calibrate on validation mixtures, then select the resulting `best-calibrated.pt` in the app. The app never uses random five-slot weights as a working separator.
