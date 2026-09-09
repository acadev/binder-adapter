"""In-process StructBioReasoner scoring backend (Tier 2).

Same ObjectiveReport contract as SubprocessScoringBackend, but calls the shared
scoring core (``scoring.py``) directly in the host interpreter instead of
spawning a subprocess per shard. Use this when the host already lives in an
environment that can import StructBioReasoner and you want to avoid the
per-shard process-spawn + JSON round-trip.

Because both backends funnel through ``scoring.score_shard``, an in-process
report and a subprocess report for the same candidate are identical.

Trade-off vs. Tier 1: no process isolation. A hard crash in an SBR dependency
(segfault, C-extension abort) takes the host down with it, and one shard's heavy
imports stay resident in the host. Prefer the subprocess backend for untrusted
or crash-prone scoring; prefer this for speed in a known-good environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from ..schemas import ObjectiveReport
from . import scoring
from .shard_io import build_shard_input, shard_dir


@dataclass(frozen=True)
class InProcessBackendConfig:
    """Tier-2 configuration.

    sbr_root: checkout of StructBioReasoner (pydantic_refactor) placed on
              sys.path of THIS interpreter.
    artifact_dir: if set, each shard writes shard_input.json + objective_reports.json
                  there, matching the subprocess backend's audit trail.
    """

    sbr_root: str
    artifact_dir: Optional[str] = None
    specificity_reference_sequences: list[str] = field(default_factory=list)


class InProcessScoringBackend:
    """Scores shards by importing SBR into the host process."""

    def __init__(self, config: InProcessBackendConfig):
        self.config = config
        # Put SBR on sys.path and probe once; capabilities are stable for the
        # life of the process.
        scoring.bootstrap_sbr_path(config.sbr_root)
        self._caps = scoring.probe_capabilities()

    @property
    def capabilities(self) -> dict[str, Any]:
        return scoring.capabilities_summary(self._caps)

    def score_candidates_batch(
        self,
        campaign_context: dict[str, Any],
        target_id: str,
        candidates_batch: List[Any],
        run_id: str,
        shard_id: int = 0,
    ) -> tuple[List[ObjectiveReport], dict[str, Any]]:
        """Score one shard. Returns (reports, metadata). Never spawns a process."""
        if not candidates_batch:
            return [], {"ok": True, "reports": 0, "note": "empty shard"}

        payload = build_shard_input(
            run_id=run_id,
            target_id=target_id,
            candidates=candidates_batch,
            campaign_context=campaign_context,
            extra_reference_sequences=self.config.specificity_reference_sequences,
        )

        d = shard_dir(self.config.artifact_dir, run_id, target_id, shard_id)
        if self.config.artifact_dir:
            (d / "shard_input.json").write_text(json.dumps(payload, indent=2))

        raw = scoring.score_shard(payload, self._caps)

        if self.config.artifact_dir:
            (d / "objective_reports.json").write_text(json.dumps(raw, indent=2))

        reports = [ObjectiveReport.from_dict(r) for r in raw.get("reports", [])]
        meta = {
            "ok": bool(raw.get("ok")),
            "exit_code": 0 if raw.get("ok") else 1,
            "capabilities": raw.get("capabilities", {}),
            "worker_errors": raw.get("worker_errors", []),
            "shard_dir": str(d),
            "objective_version": raw.get("objective_version"),
            "stderr_tail": "",
            "backend": "in_process",
        }
        return reports, meta
