#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
set -euo pipefail
ROOT=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/wan-teacache-validation
REPO=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/vllm-omni-cfg-four-gpu
PY=$ROOT/environment/venv/bin/python
OUT=$ROOT/four-gpu-$SLURM_JOB_ID
mkdir -p "$OUT"
export PYTHONPATH="$REPO:$ROOT" VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_TARGET_DEVICE=cuda PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TMPDIR=/tmp/wan-test-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
cd "$REPO"
git rev-parse HEAD > "$OUT/commit.txt"
nvidia-smi > "$OUT/gpus.txt"
nvidia-smi topo -m > "$OUT/topology.txt"
"$PY" -m pip freeze > "$OUT/requirements.txt"
"$PY" -c 'import torch,vllm;print(torch.__version__,vllm.__version__);assert torch.cuda.device_count()==4' > "$OUT/runtime.txt"
set +e
timeout --kill-after=30s 20m "$PY" -m pytest -v -s tests/diffusion/distributed/test_pipeline_parallel.py -m 'full_model and cuda and B200 and cards_4' --run-level=full_model --junitxml="$OUT/pp.xml" > "$OUT/pp.log" 2>&1
first=$?
timeout --kill-after=30s 20m "$PY" -m pytest -v -s "$ROOT/test_four_gpu_flags.py" --junitxml="$OUT/flags.xml" > "$OUT/flags.log" 2>&1
second=$?
printf '{"pp_exit":%d,"flags_exit":%d}\n' "$first" "$second" > "$OUT/status.json"
[ "$first" -eq 0 ] && [ "$second" -eq 0 ]
