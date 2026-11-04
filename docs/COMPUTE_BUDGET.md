# Remaining free-compute budget

The November 3 continuation starts after session 2: step 5,527, best validation checkpoint step 4,000. Kaggle reported 17.62 GPU hours and 20 TPU hours remaining. The user requested at least one hour untouched in each pool. Paid compute and future quota refreshes are outside this run.

`scripts/finish_kaggle_budget.py` owns the bounded continuation. It rechecks the actual GPU balance before each submission, keeps a separate fixed allowance, and includes a ten-minute platform-accounting margin. Each notebook receives a hard Kaggle timeout; its internal training deadline includes package setup and mounting time. The first continuation conservatively reserves two GPU-hours per wall hour until a completed run establishes the actual accounting rate. No mutation is automatically retried after an ambiguous submission response.

Intermediate jobs run for at most six hours. A final job receives at most four hours, including one hour reserved for validation-only threshold calibration and automatic-count held-out evaluation, plus saving time. A separate 100-update dual-T4 benchmark preserves the effective batch of eight and discards its updates. A failed/slow benchmark retains the existing single-T4 execution path. It never replaces the resumed optimizer, data sequence, or selected checkpoint.

Training ends at the first applicable limit: 20,000 steps, five stale validation checks, or the allocated quota deadline. The best validation checkpoint is selected before test evaluation. Quota completion may occur below 20,000 steps and is recorded as `quota_budget`; this is not a claim of convergence or a passed quality gate. Test failures must be reported rather than buying more compute or silently weakening the targets.

The final stage saves automatic-count metrics, a compact inference checkpoint, and deterministic four/five-speaker listening examples. A known-count diagnostic runs only if time remains after the required report. The local controller downloads results and installs the research checkpoint. Listening review remains required before a validated V1 claim.

A separate TPU compatibility probe has a 20-minute hard timeout and does not produce training weights. The PyTorch CUDA implementation cannot be credited with TPU throughput before its waveform transforms, loss assignment, numerical agreement, and backward pass have been tested. Unneeded TPU quota remains unused.

The controller needs this computer awake and online to launch subsequent stages. An already-submitted Kaggle notebook retains its hard runtime limit if the local controller stops. Restart with the identical saved policy and notebook snapshot; `STOP` prevents additional submissions, but does not cancel an active cloud notebook.
