#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=overflow
#SBATCH --account=wrd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:45:00
set -euo pipefail
ROOT=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/wan-teacache-validation
REPO=/data/wrd/projects/students/wenbo.ji/egocentric_video_generation/vllm-omni-wan-acceptance
export PATH="$ROOT/environment/venv/bin:$PATH"
export WAN_VALIDATION_ROOT="$ROOT/frozen-pp$1"
export PYTHONPATH="$REPO/tools/wan_teacache_validation:$REPO" VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_TARGET_DEVICE=cuda PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
export TMPDIR=/tmp/wan-campaign-$SLURM_JOB_ID
mkdir -p "$TMPDIR"
cd "$REPO"
OUT="$ROOT/validation-$SLURM_JOB_ID"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt"
git diff > "$OUT/worktree.patch"
nvidia-smi > "$OUT/gpus.txt"
nvidia-smi topo -m > "$OUT/topology.txt"
printenv CUDA_VISIBLE_DEVICES SLURM_JOB_ID SLURM_NODELIST > "$OUT/allocation.txt" || true
python -m pip freeze > "$OUT/requirements.txt"
python "$REPO/tools/wan_teacache_validation/campaign.py" --phase validate --pp "$1" --cfg "$2" --root "$OUT"
python "$REPO/tools/wan_teacache_validation/audit.py" "$OUT/held-out" --pp "$1" --cfg "$2"
