# Wan TeaCache validation

These scripts are an experimental calibration harness, not a production-quality
claim. The traced cache path additionally clamps negative polynomial predictions
and rejects distances beyond the calibration range. Until those policies and
stage-specific profiles are integrated into the production backend and validated,
a passing harness run alone does not qualify the production backend.

## Inputs

Use the model and exact revision in `model.json`. Download its Diffusers snapshot
to `$WAN_VALIDATION_ROOT/model`; copy `prompts.json` into that directory. The
calibration split has 24 prompts with seeds 17 and 29. The held-out split has 12
separate prompts with seeds 101, 202 and 303. Do not tune on the held-out split.
Full runs use BF16, 512x512, 17 frames, 50 steps and guidance scale 5.0.

Install the repository's dependencies with vLLM 0.31.0, then pytest, numpy,
scikit-image, imageio and imageio-ffmpeg. Preserve an exact `pip freeze`, Python,
Torch/CUDA versions, GPU topology, source SHA, commands and scheduler job ID for
each run. The validation environment used Torch 2.13.0+cu130 and Python 3.12.13.

```bash
export WAN_VALIDATION_ROOT=/shared/path/to/validation
export PATH=/shared/path/to/environment/bin:$PATH
export PYTHONPATH="$PWD/tools/wan_teacache_validation:$PWD"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=cuda
# Preserve the scheduler's CUDA_VISIBLE_DEVICES.
python tools/wan_teacache_validation/campaign.py --phase calibrate --pp 1 --cfg 1 --root "$WAN_VALIDATION_ROOT/pp1-calibration"
python tools/wan_teacache_validation/campaign.py --phase calibrate --pp 2 --cfg 1 --root "$WAN_VALIDATION_ROOT/pp2-calibration"
```

Each calibration first records full-compute, branch-local input/residual changes,
then fits one quartic per PP stage. Threshold candidates use calibration prompts
only. `selection-ppN.json` records the chosen threshold. Run held-out comparisons
for each combination of PP and CFG in {1, 2} after selecting thresholds:

```bash
python tools/wan_teacache_validation/campaign.py --phase validate --pp 2 --cfg 2 --root "$WAN_VALIDATION_ROOT/pp2-cfg2-validation"
python tools/wan_teacache_validation/audit.py "$WAN_VALIDATION_ROOT/pp2-cfg2-validation/held-out" --pp 2 --cfg 2
```

The generator alternates native and cached requests in a single loaded model,
warms up both paths, and excludes loading, warmup and output serialization from
request timing. It saves float output arrays, MP4, first/middle/last frames, final
latents and real-hook per-rank decisions. `audit.py` checks first-call computation,
CFG rank/branch mapping, finite paired latents and actual skips. Trace I/O remains
inside request timing. Inspect videos as well as automated measurements.

`comparison.json` records mean frame SSIM, worst video mean SSIM, mean temporal
change error relative to native temporal change, and cached/native latency ratio.
Acceptance requires respectively >=0.95, >=0.90, <=0.10 and <=0.90. A failed gate
must remain visible; unsuccessful measurements must not be described as production
acceptance. The `smoke` split uses 5 frames at 256x256 and is only a startup test.
