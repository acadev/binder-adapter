# Deploying binder-adapter at scale (portable HPC)

This directory holds the deployment artifacts for running binder-adapter scoring
across supercomputers via Academy + Parsl, packaged in an Apptainer container.

The design goal is **portability**: the scoring code, the Academy `ScoringAgent`,
and the batch entrypoint never change between machines. You only pick a Parsl
site config (`examples/parsl_configs.py`, selected with `--site`) and, on a real
cluster, run inside the container.

## Files

| file | purpose |
|------|---------|
| `../apptainer.def` | reproducible container: OpenMM MD stack + academy-py + parsl + the adapter, in an isolated conda env |
| `slurm_smoke_test.py` | minimal "does scale-out work here" check: loads the site Parsl config, places a task on the allocation, scores one shard through the SBR stack |
| `leonardo_smoke.sbatch` | CINECA Leonardo (Booster, GPU) sbatch that runs the smoke test inside the container |

## 1. Build the container

On a machine with Apptainer and fakeroot/sudo (a login node usually works):

```bash
cd /path/to/binder-adapter
apptainer build binder-adapter.sif apptainer.def
```

The build runs a `%test` self-check; if the stack doesn't import, the build
fails. Pinned versions inside the image mirror a known-good env: academy-py
1.0.0, parsl 2026.2.9, openmm 8.3.1, pdbfixer 1.12.0, mdtraj 1.10.0, numpy
1.26.4, openai 2.7.2, Python 3.12.

**Why a container:** `academy-py>=1.0` pulls `globus-sdk>=4`, which can clash
with other packages (e.g. proxystore) in a shared base env. Isolating the whole
stack in an image makes runs reproducible across facilities and immune to module
drift on thousands of nodes.

**What stays outside the image** (bind-mounted at run time, because they are
evolving research checkouts, not release deps):
- StructBioReasoner -> `/opt/StructBioReasoner` (`BINDER_ADAPTER_SBR_ROOT`)
- Jnana (optional)   -> `/opt/Jnana` (`BINDER_ADAPTER_JNANA_ROOT`)
- scratch/artifacts  -> `/scratch`

## 2. Run the SLURM smoke test (do this FIRST on a new machine)

Before committing an allocation to a real campaign, prove the scale-out path
works on the target machine:

```bash
# on Leonardo, after building the .sif and checking out SBR:
export CONTAINER=$HOME/binder-adapter.sif
export BINDER_ADAPTER_SBR_ROOT=$HOME/StructBioReasoner
export SCRATCH_DIR=$SCRATCH/binder_adapter_smoke
sbatch --account=<YOUR_PROJECT> deploy/leonardo_smoke.sbatch
```

The job prints `RESULT: PASS` when:
1. the container self-test passes (stack imports on the compute node),
2. Parsl places a task on the allocation (prints the node hostname), and
3. the Academy backend scores a shard through SBR (returns real objectives).

Non-zero exit = failure, so you can gate automation on it.

**Laptop / login-node dry run** (same code path, no scheduler):

```bash
PYTHONPATH=.:$BINDER_ADAPTER_SBR_ROOT \
  python deploy/slurm_smoke_test.py --site local --sbr-root $BINDER_ADAPTER_SBR_ROOT
```

This is exactly what the containerized job runs; it has been validated to
produce `RESULT: PASS` (Parsl task placed, shard scored with developability +
specificity active).

## 3. Run a real campaign

Generation is decoupled from scoring (compute nodes usually have no internet):
produce `candidates.jsonl` upstream (Jnana+ARGO on a login/service node, or a
local vLLM), then score at scale:

```bash
apptainer run --nv \
    --bind $HOME/StructBioReasoner:/opt/StructBioReasoner \
    --bind $SCRATCH:/scratch \
    binder-adapter.sif \
        --candidates /scratch/candidates.jsonl \
        --framework SQETFSDLWKLLPEN --target MDM2 \
        --site leonardo --md --md-platform CUDA \
        --artifact-dir /scratch/runs --out /scratch/rankings.json
```

`--nv` exposes the host NVIDIA driver for GPU-resident MM-GBSA MD. With
`--site leonardo`, the driver runs on the login node and Parsl submits blocks to
the Booster partition; each A100 gets one MD shard (`available_accelerators=4`).

## Porting to another machine

Add one factory to `examples/parsl_configs.py` returning a `parsl.config.Config`
with your scheduler's provider (Slurm/PBSPro/LSF/Cobalt/Flux), register it in
`SITE_CONFIGS`, and use `--site <name>`. A `polaris` (ALCF, PBSPro) example is
included alongside `leonardo` to show the pattern is scheduler-agnostic. Nothing
else changes.

### Checklist for a new site
- [ ] `available_accelerators` matches GPUs/node; `max_workers_per_node` set
- [ ] `worker_init` activates the SAME container/env on workers as the driver
- [ ] `--artifact-dir` points at `$SCRATCH`/`$FAST`, never `$HOME`
- [ ] validate ΔG parity CPU-vs-CUDA on a known complex before large MD runs
- [ ] run the smoke test and confirm `RESULT: PASS`
