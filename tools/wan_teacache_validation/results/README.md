# Wan TeaCache validation results (2026-10-10)

**Experimental draft; production acceptance has not passed.** No default Wan
TeaCache profile is enabled. Eighteen completed calibration candidates failed at
least one quality/performance gate. The independent evaluation uses frozen,
calibration-selected diagnostic profiles and does not tune on held-out prompts.
PP=2 held-out jobs are still pending; their scheduling is not a passing result.

## Sources and configuration

- Original #8475 four-GPU validation: `7e88fc3eb2685d84e5c04d9ebe049a88f7151106`.
- Wan independent evaluation: `ef6379755f1f12a64a41336efa70b62e43b933f6`.
  Subsequent changes to this branch are reports and a stricter post-run audit.
- Dependencies: #8475 plus #8483 explicit CFG branch identification, merged at
  `e572a8f6`. Wan does not implement a second counter-parity workaround.
- Model: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, revision
  `0fad780a534b6463e45facd96134c9f345acfa5b` (real pretrained weights).
- Python 3.12.13, Torch 2.13.0+cu130, vLLM 0.31.0. See
  [dependency lock](../requirements.lock.txt) and each run's provenance.
- BF16, TP=SP=1, 512x512, 17 frames, 50 steps, guidance 5.0.
  [Prompts and seeds](../prompts.json): calibration 24x2, held-out 12x3.
- Slurm: `overflow`, account `wrd`, job name `test`, one node, four B200,
  32 CPUs, 256 GB, at most eight hours. GPU visibility was scheduler-assigned.

## Original PR four-GPU regressions

Job **1095857**, on the original #8475 SHA: **4 passed, 0 failed, 0 skipped**.
Two existing PP prediction/scheduler tests and two FP32/BF16 flag-transition
checks ran with PP=2/CFG=2. All four rank identities were verified; the flag
followed CFG on -> off -> on before prediction and the last-stage result matched
the single-GPU baseline. FP32 tolerance was 1e-5, BF16 1e-2.
These are numerical pipeline tests, not full pretrained video acceptance.
See [JUnit and allocation evidence](original-8475/). Results were also posted in
[#8475](https://github.com/vllm-project/vllm-omni/pull/8475#issuecomment-6096951824).

## Feature regressions and full-compute equivalence

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Merged CFG/TeaCache CPU regressions | 34 | 0 | 0 |
| PP CPU suite | 29 | 0 | 0 |
| Wan pipeline diffuse/decode CPU suite | 22 | 0 | 0 |
| Latest unsupported Wan variant guards | 4 | 0 | 0 |
| Actual small Wan hook, FP32/BF16 x full/reuse | 4 | 0 | 0 |
| Latest explicit PP branch stamping regression | 1 | 0 | 0 |

Suites overlap and were run at different implementation stages; do not sum these
as unique cases or attribute all of them to the final report commit. [JUnit](junit/)
retains individual test names and timestamps. Ruff passes on the changed files.
Mypy is not clean: 12 pre-existing PP mixin errors remain, versus 14 in the merged
dependency baseline. The new Wan files and tests pass the targeted type checks.

All four full-pretrained topologies (PP1/CFG1, PP1/CFG2, PP2/CFG1, PP2/CFG2)
produced identical decoded arrays with caching disabled versus a real mounted
hook forced to compute every block: **maximum pixel difference 0**. These smoke
runs used 256x256, 5 frames and 6 steps, with jobs 1095891/1095892 for the distributed
cases. This proves integration/equivalence at that size, not production quality.
See [smoke comparisons](smoke/); the raw run logs record their earlier source SHAs.

A pre-existing Wan PP VAE-output bug (`None.dim()`) was reproduced on #8475 both
through the actual CPU pipeline forward and a real pretrained GPU startup
(job 1096049). The feature branch handles the non-owner `(None,)` output.
Intermittent multiprocessing `SemLock._rebuild` startup errors also occurred on
both baseline and feature runs. Those failed attempts are not passing tests.

## Calibration and frozen profiles

Each PP topology collected 48 videos with full computation. Each local stage
contributed 4,704 consecutive-step input/residual pairs across both branches.
Jobs: PP1 **1095869**, PP2 **1095893**. Fit errors, input ranges, polynomial
coefficients, selection evidence and checksums are in [profiles](profiles/).
The [18-candidate calibration table](calibration.json) includes all completed
coarse, fine-threshold and warmup strategies (12 PP1, 6 PP2).
A PP2 warmup-35 startup failure and a canceled queued retry are not candidates.

Frozen diagnostics: PP1 threshold 0.07, no forced warmup; PP2 threshold 0.1,
30 full-compute warmup steps. Both are explicitly `diagnostic_only: true`.
The harness uses stage-specific fitted polynomials, clamps negative predictions,
and forces computation above the largest observed input distance. These extra
policies are not production backend defaults, so harness evidence alone cannot
qualify the production path.

## Independent held-out evaluation

Each completed row represents 36 paired native/cached requests in one loaded
model, with alternating order and warmup excluded. Quality is measured on float
RGB frames before MP4 compression. Lower latency ratio is better.

| PP / CFG | Job | Mean SSIM | Worst video SSIM | Temporal error | Cached/native latency | Gate result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 / 1 | 1096032 | 0.988863 | 0.971094 | 0.163825 | 1.027440 | 2 pass / 2 fail |
| 1 / 2 | 1096033 | 0.988863 | 0.971094 | 0.163825 | 1.044654 | 2 pass / 2 fail |
| 2 / 1 | 1096100 | pending | pending | pending | pending | not evaluated |
| 2 / 2 | 1096101 | pending | pending | pending | pending | not evaluated |
| Required | | >=0.95 | >=0.90 | <=0.10 | <=0.90 | all gates |

Each completed PP1 run recorded 1,690 full local-stack calls and 110 cache hits
per CFG branch across 36 requests. Every request began at step zero with a full
computation. Branch states were distinct, final paired latents were finite, and
actual block-stack calls were skipped. [Machine-readable results](held-out.json)
include per-video SSIM, temporal error, latent max-absolute/RMSE/relative-L2 error
and per-stage/branch counts. Each run also has its own four-gate JUnit report.

The temporal metric is the mean absolute difference between cached and native
frame differences, normalized by native mean frame difference. It is not merely
a motion-magnitude ratio and does not alone prove visible flicker. Its exact
formula and tracing/timing limitations are in the [protocol](../README.md).

## Artifact locations

Full logs, rank traces, GPU topology, Slurm allocation, dependency snapshots,
latents, float decoded arrays, MP4s and frames 0/8/16 are preserved under:

```text
/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/wan-teacache-validation/
  four-gpu-1095857/
  campaign-<job>/
  validation-<job>/held-out/{none,cache,trace}/
```

[Video and frame samples](video-samples/) include a real PP2/CFG1 calibration
run with actual cache reuse and a PP1/CFG2 held-out run. They are explicitly
labeled by split and must not be treated as four-card held-out acceptance.

Large weights, latent tensors and float frame arrays are intentionally kept in the
shared validation directory. Committed reports retain source provenance and
numerical results; pending jobs must finish before the four-topology held-out
matrix can be considered complete.
