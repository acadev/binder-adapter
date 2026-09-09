#!/bin/bash
#SBATCH --job-name=binder-adapter-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=binder-adapter-smoke-%j.out
#SBATCH --error=binder-adapter-smoke-%j.err
# NOTE: set your allocation before submitting, e.g.:
#   sbatch --account=<PROJECT> deploy/leonardo_smoke.sbatch
# ============================================================================
# Real SLURM smoke test for binder-adapter on CINECA Leonardo (Booster).
#
# What it proves, end to end, on an actual GPU node:
#   1. the Apptainer image self-test passes (stack imports on the node),
#   2. Parsl can place a task on the allocation,
#   3. the Academy scoring backend scores a shard through the SBR stack.
#
# It runs the containerized smoke test with --site local, because inside a
# single allocated node you want Parsl's LocalProvider to use the node you are
# already on (not submit a NESTED SLURM job). The `leonardo` site config is for
# the driver that runs on the LOGIN node and submits blocks to compute — that is
# exercised separately by the full campaign, not this single-node smoke.
#
# Prereqs on Leonardo:
#   - binder-adapter.sif built (see apptainer.def) at $CONTAINER
#   - StructBioReasoner checkout at $SBR_ROOT
#   - a writable scratch dir at $SCRATCH_DIR
# ============================================================================
set -euo pipefail

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
APPTAINER=$(command -v apptainer || command -v singularity)

# ---- paths (override via environment before sbatch) ------------------------
CONTAINER="${CONTAINER:-$HOME/binder-adapter.sif}"
SBR_ROOT="${BINDER_ADAPTER_SBR_ROOT:-$HOME/StructBioReasoner}"
SCRATCH_DIR="${SCRATCH_DIR:-${SCRATCH:-$HOME}/binder_adapter_smoke}"
REPO_IN_IMG="/opt/binder-adapter"

mkdir -p "$SCRATCH_DIR"
echo "container : $CONTAINER"
echo "sbr_root  : $SBR_ROOT"
echo "scratch   : $SCRATCH_DIR"
echo "node      : $(hostname)  gpus: ${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi -L || echo "WARN: nvidia-smi not available"

# ---- 0. container self-test (fails fast if the image is broken) -----------
echo "=== [0] container self-test ==="
"$APPTAINER" test --nv "$CONTAINER"

# ---- 1-3. run the smoke test INSIDE the container on this GPU node ---------
echo "=== [1-3] scale-out smoke (Parsl task + Academy scoring shard) ==="
"$APPTAINER" exec --nv \
    --bind "$SBR_ROOT:/opt/StructBioReasoner" \
    --bind "$SCRATCH_DIR:/scratch" \
    "$CONTAINER" \
    python "$REPO_IN_IMG/deploy/slurm_smoke_test.py" \
        --site local \
        --sbr-root /opt/StructBioReasoner \
        --artifact-dir /scratch/smoke \
        --n 2

echo "=== smoke test exit code: $? ==="
echo "If RESULT: PASS appeared above, the scale-out path works on Leonardo."
