# Research basis

| Work | Use |
|---|---|
| [TF-Locoformer (2024)](https://arxiv.org/abs/2408.03440) | Primary time-frequency backbone; source and weight licensing verified. |
| [Hungarian PIT (2021)](https://arxiv.org/abs/2104.08955) | Polynomial output/reference assignment; higher-count research precedent. |
| [SepTDA (2024)](https://arxiv.org/abs/2401.12473) | Attractor-based counting/separation comparator. |
| [A2PIT (2020)](https://arxiv.org/abs/2003.12326) | Variable source-count learning. VoiceSep uses explicit silence-safe losses, not A2PIT's mixture-copy targets. |
| [OR-PIT (2019)](https://arxiv.org/abs/1904.03065) | Recursive extraction scaling baseline. |
| [DExFormer (2025)](https://www.mdpi.com/1424-8220/25/16/4905) | Later learned extraction and stopping experiments. |
| [Separate and Diffuse (ICLR 2024)](https://openreview.net/pdf?id=UXALv0lJZS) | Later refinement experiment; evaluate content fidelity and cost. |

VoiceSep's five-slot/activity extension is a proposed adaptation, not a published TF-Locoformer result. Source-count prediction, separation quality, and identity tracking are different measurements. A five-slot output tensor alone demonstrates none of them.

Primary novel contribution candidate: quality-aware bilingual speaker analysis. Compare age/affect errors on original and separated speech, then test whether cleaner-window selection and abstention improve calibrated reliability. Feature panels alone are not a research novelty claim.
