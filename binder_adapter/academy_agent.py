"""Academy agent wrapping the binder-adapter scoring pipeline.

This is the portable orchestration layer. A ``ScoringAgent`` is a stateful
Academy agent that:

* holds the campaign context + backend configuration as state,
* exposes ``score_and_select`` and ``score_shards`` as ``@action`` methods that a
  user script (or a peer coordinator agent) can invoke remotely,
* dispatches the actual scoring through the Parsl-backed
  :class:`~binder_adapter.sbr_backends.academy_backend.AcademyScoringBackend`.

Why an agent at all (vs. just calling Parsl directly)? Two reasons that matter at
scale and across sites:

1. Placement portability. Academy separates the *exchange* (how agents message)
   from the *executor* (where an agent runs). The same ``ScoringAgent`` can run
   in-process on a login node today, or be launched onto a remote Globus Compute
   endpoint at another facility tomorrow, with no change to this class.
2. Statefulness. A campaign is a long-lived feedback loop (generate -> score ->
   select -> resample), not a one-shot task graph. An agent keeps campaign state
   (capabilities probe, budgets, run id) between calls, which is exactly the
   pattern Academy targets and plain task-graph systems do not.

Run locally with ``LocalExchangeFactory`` + a ``ThreadPoolExecutor``; the
scoring fan-out still uses Parsl underneath, so you get real parallelism even in
the local development path.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from academy.agent import Agent, action

    _ACADEMY_OK = True
except Exception:  # pragma: no cover - env dependent
    _ACADEMY_OK = False

    class Agent:  # type: ignore
        """Fallback so the module imports even without academy installed."""

    def action(fn):  # type: ignore
        return fn


from .adapter_core import AdapterConfig, BinderAdapterCore
from .sbr_backends.academy_backend import AcademyBackendConfig, AcademyScoringBackend


class ScoringAgent(Agent):
    """Stateful Academy agent that scores + selects binder candidates at scale.

    Parameters
    ----------
    sbr_root:
        StructBioReasoner checkout importable on the workers.
    parsl_config_factory:
        Optional zero-arg callable returning a parsl Config. Defaults to a local
        HighThroughputExecutor. Pass a SLURM/PBS/Flux factory to target a
        cluster (see ``examples/parsl_configs.py``).
    shard_size, max_survivors, mutation budget:
        forwarded to :class:`AdapterConfig`.
    """

    def __init__(
        self,
        sbr_root: str,
        parsl_config_factory=None,
        shard_size: int = 8,
        artifact_dir: Optional[str] = None,
        min_mutated_fraction: float = 0.20,
        max_mutated_fraction: float = 0.50,
        max_survivors: int = 25,
        specificity_reference_sequences: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.sbr_root = sbr_root
        self.shard_size = shard_size
        self.artifact_dir = artifact_dir
        self.min_mutated_fraction = min_mutated_fraction
        self.max_mutated_fraction = max_mutated_fraction
        self.max_survivors = max_survivors
        self._backend = AcademyScoringBackend(
            AcademyBackendConfig(
                sbr_root=sbr_root,
                parsl_config_factory=parsl_config_factory,
                artifact_dir=artifact_dir,
                specificity_reference_sequences=specificity_reference_sequences or [],
            )
        )

    # -- actions ----------------------------------------------------------

    @action
    async def score_and_select(
        self,
        hypotheses: list[dict[str, Any]],
        campaign_context: dict[str, Any],
        run_id: str,
        target_ids: Optional[list[str]] = None,
        max_survivors: Optional[int] = None,
    ) -> dict[str, Any]:
        """Full campaign round using the Parsl fan-out backend.

        Delegates extraction/gating/selection to BinderAdapterCore but routes the
        SBR scoring through the Academy/Parsl backend so shards run concurrently
        across the site's compute.
        """
        core = self._make_core()
        return core.score_and_select(
            hypotheses=hypotheses,
            campaign_context=campaign_context,
            run_id=run_id,
            target_ids=target_ids,
            max_survivors=max_survivors or self.max_survivors,
        )

    @action
    async def capabilities(self) -> dict[str, Any]:
        """Report what SBR objectives are computable on the workers."""
        from .sbr_backends import scoring

        scoring.bootstrap_sbr_path(self.sbr_root)
        return scoring.capabilities_summary(scoring.probe_capabilities())

    @action
    async def shutdown_parsl(self) -> str:
        self._backend.shutdown()
        return "parsl shut down"

    # -- internals --------------------------------------------------------

    def _make_core(self) -> BinderAdapterCore:
        """Build an adapter core whose backend is our Parsl-backed one.

        We construct the core with the in_process backend for its extraction /
        gating / selection logic, then swap in the Academy backend so scoring
        fans out. (The core calls ``backend.score_candidates_batch`` per shard;
        our backend satisfies that interface and parallelizes internally.)
        """
        core = BinderAdapterCore(
            AdapterConfig(
                sbr_root=self.sbr_root,
                backend="in_process",
                shard_size=self.shard_size,
                artifact_dir=self.artifact_dir,
                min_mutated_fraction=self.min_mutated_fraction,
                max_mutated_fraction=self.max_mutated_fraction,
                max_survivors=self.max_survivors,
            )
        )
        core.backend = self._backend  # route scoring through Parsl fan-out
        return core


def build_local_scoring_agent(sbr_root: str, **kwargs) -> ScoringAgent:
    """Convenience constructor for the local development path."""
    return ScoringAgent(sbr_root=sbr_root, **kwargs)
