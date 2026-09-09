"""Parsl site configurations for binder-adapter scale-out.

This is the ONE file you edit to move between machines. The scoring code, the
Academy agent, and the batch entrypoint never change — you only pick (or add) a
config factory here and pass it to the ScoringAgent / AcademyScoringBackend.

Each factory returns a ``parsl.config.Config``. The pattern is always the same:
a ``HighThroughputExecutor`` (which gives you one lightweight worker pool that
Parsl feeds) wrapped around a *provider* that knows how to get nodes from the
site's scheduler (Local, Slurm, PBSPro, LSF, Cobalt, Flux, ...).

Usage:

    from examples.parsl_configs import leonardo_booster_config
    agent = ScoringAgent(sbr_root=..., parsl_config_factory=leonardo_booster_config)

Notes that apply on real machines:
  * Set ``worker_init`` to activate the SAME environment (conda/venv/container)
    on the workers that has openmm+pdbfixer+StructBioReasoner. Mismatched worker
    envs are the #1 cause of "works on login node, fails on compute".
  * Point artifact_dir (in the agent/backend) at $SCRATCH / $FAST, never $HOME.
  * For GPU MD, request GPU nodes in the provider and set the MD platform to
    "CUDA" in md_config; pin one shard per GPU via the executor's worker count.
"""

from __future__ import annotations

import os


# ---------------------------------------------------------------------------
# 1. LOCAL — laptop / login-node development. The portability baseline.
# ---------------------------------------------------------------------------
def local_config(max_workers: int = 4):
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    return Config(
        executors=[
            HighThroughputExecutor(
                label="local",
                max_workers_per_node=max_workers,
                provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
            )
        ],
    )


# ---------------------------------------------------------------------------
# 2. CINECA LEONARDO — Booster partition (4x A100-64GB per node), SLURM.
#    GPU-resident MM-GBSA MD: one worker per GPU.
# ---------------------------------------------------------------------------
def leonardo_booster_config(
    nodes: int = 1,
    walltime: str = "01:00:00",
    account: str = os.environ.get("SLURM_ACCOUNT", "REPLACE_ME"),
    env_activate: str = os.environ.get(
        "BINDER_ADAPTER_WORKER_INIT",
        "module load cuda; source $HOME/binder-adapter-env/bin/activate",
    ),
):
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SrunLauncher
    from parsl.providers import SlurmProvider

    return Config(
        executors=[
            HighThroughputExecutor(
                label="leonardo_booster",
                # 4 A100s per Booster node -> 4 concurrent MD shards per node.
                max_workers_per_node=4,
                available_accelerators=4,  # pins one worker per GPU
                provider=SlurmProvider(
                    partition="boost_usr_prod",
                    account=account,
                    nodes_per_block=nodes,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    walltime=walltime,
                    scheduler_options="#SBATCH --gres=gpu:4",
                    worker_init=env_activate,
                    launcher=SrunLauncher(),
                ),
            )
        ],
    )


# ---------------------------------------------------------------------------
# 3. ALCF POLARIS — example GPU config (4x A100 per node), PBSPro.
#    Shows the pattern is scheduler-agnostic: only the provider changes.
# ---------------------------------------------------------------------------
def polaris_config(
    nodes: int = 1,
    walltime: str = "01:00:00",
    account: str = os.environ.get("PBS_ACCOUNT", "REPLACE_ME"),
    queue: str = "debug",
    env_activate: str = os.environ.get(
        "BINDER_ADAPTER_WORKER_INIT",
        "module use /soft/modulefiles; module load conda; conda activate binder-adapter",
    ),
):
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import MpiExecLauncher
    from parsl.providers import PBSProProvider

    return Config(
        executors=[
            HighThroughputExecutor(
                label="polaris",
                max_workers_per_node=4,
                available_accelerators=4,
                provider=PBSProProvider(
                    account=account,
                    queue=queue,
                    nodes_per_block=nodes,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    walltime=walltime,
                    worker_init=env_activate,
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                    ),
                    cpus_per_node=64,
                ),
            )
        ],
    )


# Registry so the batch entrypoint can pick a config by name (--site).
SITE_CONFIGS = {
    "local": local_config,
    "leonardo": leonardo_booster_config,
    "polaris": polaris_config,
}
