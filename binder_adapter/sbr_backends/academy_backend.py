"""Academy + Parsl scoring backend for portable HPC scale-out.

Why this backend exists
-----------------------
The subprocess/in-process backends score shards **sequentially** in one host
process. That is fine for a laptop but wastes a supercomputer. This backend
fans the same shards out across whatever compute a site provides, WITHOUT tying
the code to one scheduler.

Portability comes from `Academy <https://academy-agents.org>`_ + Parsl:

* The scoring work is expressed once as a Parsl ``@python_app`` (``_score_shard_app``)
  that calls the SAME shared scoring core (`scoring.score_shard`) the other
  backends use — so a report produced here is byte-identical to an in-process or
  subprocess report for the same candidate.
* WHERE that app runs is decided by a Parsl ``Config`` (an executor + a provider).
  A ``LocalProvider`` runs it on your laptop; a ``SlurmProvider`` (or PBS, LSF,
  Cobalt, Flux ...) runs it across a cluster. Swapping the provider is the ONLY
  change needed to move between CINECA Leonardo, ALCF, NERSC, etc.
* An Academy ``ScoringAgent`` holds campaign state and exposes a ``score_shards``
  ``@action``. Because Academy separates the *exchange* (messaging) from the
  *executor* (placement), the very same agent can run locally today and be
  launched onto a remote endpoint (e.g. via Globus Compute) tomorrow with no
  change to the agent code.

This module degrades gracefully: if ``academy``/``parsl`` are not installed it
raises a clear error at construction, and the rest of the adapter keeps working
with the local backends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

from ..schemas import CandidateArtifact, ObjectiveReport
from .shard_io import build_shard_input, shard_dir


class AcademyBackendUnavailable(RuntimeError):
    """Raised when academy/parsl are not importable."""


def academy_available() -> tuple[bool, str]:
    try:
        import academy  # noqa: F401
        import parsl  # noqa: F401

        return True, ""
    except Exception as exc:  # pragma: no cover - env dependent
        return False, str(exc)


# ---------------------------------------------------------------------------
# The unit of parallel work: score ONE shard from its serialized input.
#
# This is a module-level function (picklable) that takes the exact same
# shard_input dict the subprocess worker consumes and returns the exact same
# raw report dict scoring.score_shard produces. Everything the task needs
# travels inside `shard_input` + `sbr_root`, so it is self-contained and can run
# on a remote worker with no shared state.
# ---------------------------------------------------------------------------
def score_shard_payload(shard_input: dict[str, Any], sbr_root: str) -> dict[str, Any]:
    """Pure function run on a (possibly remote) worker. No adapter state needed."""
    # Imports are inside the function so the remote worker resolves them in its
    # own environment (Parsl ships the function, not the whole process).
    from binder_adapter.sbr_backends import scoring

    scoring.bootstrap_sbr_path(sbr_root)
    caps = scoring.probe_capabilities()
    return scoring.score_shard(shard_input, caps)


@dataclass(frozen=True)
class AcademyBackendConfig:
    """Configuration for the Academy/Parsl fan-out backend.

    sbr_root: StructBioReasoner checkout, made importable on each worker.
    parsl_config_factory: zero-arg callable returning a parsl.config.Config.
        Defaults to a local thread/process config. To target a cluster, pass a
        factory that builds a Config with a SlurmProvider (see
        ``examples/parsl_configs.py``). This is the ONE site-specific knob.
    max_workers: hint for the default local config only.
    artifact_dir: if set, mirrors the subprocess/in-process audit trail
        (shard_input.json + objective_reports.json per shard).
    """

    sbr_root: str
    parsl_config_factory: Optional[Callable[[], Any]] = None
    max_workers: int = 4
    artifact_dir: Optional[str] = None
    specificity_reference_sequences: list[str] = field(default_factory=list)


def default_local_parsl_config(max_workers: int = 4) -> Any:
    """A laptop-friendly Parsl config: HighThroughputExecutor + LocalProvider.

    This is the portability baseline — identical agent/task code runs here and on
    a cluster; only the provider differs.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    return Config(
        executors=[
            HighThroughputExecutor(
                label="binder_adapter_local",
                max_workers_per_node=max_workers,
                provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
            )
        ],
    )


class AcademyScoringBackend:
    """Parsl-backed fan-out backend with the standard backend interface.

    Exposes ``score_candidates_batch`` (one shard) like the other backends, plus
    ``score_shards_parallel`` (many shards at once) which is where the speedup
    lives. The adapter core can call either.
    """

    def __init__(self, config: AcademyBackendConfig):
        ok, why = academy_available()
        if not ok:
            raise AcademyBackendUnavailable(
                f"academy/parsl not importable: {why}. "
                "Install with `pip install academy-py parsl` in an isolated env."
            )
        self.config = config
        self._dfk = None  # lazy Parsl data-flow kernel

    # -- Parsl lifecycle --------------------------------------------------

    def _ensure_parsl(self):
        import parsl

        if self._dfk is None:
            factory = self.config.parsl_config_factory or (
                lambda: default_local_parsl_config(self.config.max_workers)
            )
            self._dfk = parsl.load(factory())
        return self._dfk

    def shutdown(self) -> None:
        if self._dfk is not None:
            import parsl

            parsl.dfk().cleanup()
            parsl.clear()
            self._dfk = None

    # -- single-shard (standard interface, parity with other backends) ----

    def score_candidates_batch(
        self,
        campaign_context: dict[str, Any],
        target_id: str,
        candidates_batch: List[CandidateArtifact],
        run_id: str,
        shard_id: int = 0,
    ) -> tuple[List[ObjectiveReport], dict[str, Any]]:
        reports_by_shard = self.score_shards_parallel(
            campaign_context=campaign_context,
            target_id=target_id,
            shards=[candidates_batch],
            run_id=run_id,
            base_shard_id=shard_id,
        )
        return reports_by_shard[0]

    # -- many-shards (the scale-out path) ---------------------------------

    def score_shards_parallel(
        self,
        campaign_context: dict[str, Any],
        target_id: str,
        shards: List[List[CandidateArtifact]],
        run_id: str,
        base_shard_id: int = 0,
    ) -> list[tuple[List[ObjectiveReport], dict[str, Any]]]:
        """Score many shards concurrently via Parsl. Order is preserved.

        Each shard becomes one Parsl task; the site's executor/provider decides
        placement. Empty shards short-circuit without a task.
        """
        from parsl import python_app

        self._ensure_parsl()

        # Wrap the module-level pure function as a Parsl app once.
        app = python_app(score_shard_payload)

        futures = []
        payloads = []
        for i, batch in enumerate(shards):
            shard_id = base_shard_id + i
            if not batch:
                futures.append(None)
                payloads.append(None)
                continue
            payload = build_shard_input(
                run_id=run_id,
                target_id=target_id,
                candidates=batch,
                campaign_context=campaign_context,
                extra_reference_sequences=self.config.specificity_reference_sequences,
            )
            payloads.append((shard_id, payload))
            if self.config.artifact_dir:
                d = shard_dir(self.config.artifact_dir, run_id, target_id, shard_id)
                (d / "shard_input.json").write_text(json.dumps(payload, indent=2))
            futures.append(app(payload, self.config.sbr_root))

        results: list[tuple[List[ObjectiveReport], dict[str, Any]]] = []
        for fut, meta_in in zip(futures, payloads):
            if fut is None:
                results.append(([], {"ok": True, "reports": 0, "note": "empty shard"}))
                continue
            shard_id, _payload = meta_in
            try:
                raw = fut.result()
            except Exception as exc:  # a failed shard must not fake scores
                results.append(
                    (
                        [],
                        {
                            "ok": False,
                            "exit_code": 1,
                            "worker_errors": [f"{type(exc).__name__}: {exc}"],
                            "backend": "academy_parsl",
                            "shard_id": shard_id,
                        },
                    )
                )
                continue

            if self.config.artifact_dir:
                d = shard_dir(self.config.artifact_dir, run_id, target_id, shard_id)
                (d / "objective_reports.json").write_text(json.dumps(raw, indent=2))

            reports = [ObjectiveReport.from_dict(r) for r in raw.get("reports", [])]
            meta = {
                "ok": bool(raw.get("ok")),
                "exit_code": 0 if raw.get("ok") else 1,
                "capabilities": raw.get("capabilities", {}),
                "worker_errors": raw.get("worker_errors", []),
                "shard_dir": str(
                    shard_dir(self.config.artifact_dir, run_id, target_id, shard_id)
                )
                if self.config.artifact_dir
                else "",
                "objective_version": raw.get("objective_version"),
                "stderr_tail": "",
                "backend": "academy_parsl",
                "shard_id": shard_id,
            }
            results.append((reports, meta))
        return results
