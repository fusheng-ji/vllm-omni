# Real pretrained video samples

- `pp1-cfg2-held-out`: 17-frame, 512x512 held-out prompt 00, seed 101,
  job 1096033. Native and actual-cache videos, plus frames 0/8/16.
- `pp2-cfg1-calibration`: 17-frame, 512x512 calibration prompt 00, seed 17,
  job 1096035. This is real PP + sequential CFG + actual TeaCache reuse,
  **not held-out acceptance**. Native/cache video and frames are included.

Both use the pinned Wan2.1-T2V-1.3B pretrained checkpoint, BF16, 50 steps and
guidance 5. The PP2 diagnostic profile uses 30 full-compute warmup steps followed
by real threshold decisions. Neither profile passed all production gates.
These samples are selected by prompt index, not as best-looking outputs.
The separate PP2/CFG2 held-out run remains pending in the results matrix.

Manual inspection of frames 0/8/16 of the PP2 calibration sample finds visible
background speckles in its last frame in both the native and cached output.
This is not evidence that the cache alone introduced that artifact, and these
examples are not a passing visual-quality review.
