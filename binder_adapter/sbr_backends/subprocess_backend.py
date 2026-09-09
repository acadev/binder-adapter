from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from ..schemas import CandidateArtifact, ObjectiveReport
from .shard_io import build_shard_input, shard_dir

WORKER_PATH = Path(__file__).resolve().parent / "sbr_scoring_worker.py"


@dataclass(frozen=True)
class SubprocessBackendConfig:
    """Tier-1 configuration.

    sbr_root: checkout of StructBioReasoner (pydantic_refactor) placed on the
              worker's sys.path.
    python_executable: interpreter that can import the SBR deps (MDAnalysis etc).
    """

    sbr_root: str
    python_executable: str = sys.executable
    timeout_seconds: int = 3600
    artifact_dir: Optional[str] = None
    specificity_reference_sequences: list[str] = field(default_factory=list)


class SbrSubprocessUnavailable(RuntimeError):
    """Raised when the worker could not be executed at all."""


class SubprocessScoringBackend:
    """Runs the SBR scoring worker once per (target, candidate shard).

    Everything this returns traces back to StructBioReasoner code or is
    explicitly flagged unavailable. There are no synthesised scores.
    """

    def __init__(self, config: SubprocessBackendConfig):
        self.config = config

    # -- shard IO ---------------------------------------------------------

    def _shard_dir(self, run_id: str, target_id: str, shard_id: int) -> Path:
        return shard_dir(self.config.artifact_dir, run_id, target_id, shard_id)

    def _build_input(
        self,
        run_id: str,
        target_id: str,
        candidates: List[CandidateArtifact],
        campaign_context: dict[str, Any],
    ) -> dict[str, Any]:
        return build_shard_input(
            run_id=run_id,
            target_id=target_id,
            candidates=candidates,
            campaign_context=campaign_context,
            extra_reference_sequences=self.config.specificity_reference_sequences,
        )

    # -- execution --------------------------------------------------------

    def score_candidates_batch(
        self,
        campaign_context: dict[str, Any],
        target_id: str,
        candidates_batch: List[CandidateArtifact],
        run_id: str,
        shard_id: int = 0,
    ) -> tuple[List[ObjectiveReport], dict[str, Any]]:
        """Shared backend interface (mirrors InProcessScoringBackend)."""
        return self.score_candidates_batch_subprocess(
            campaign_context=campaign_context,
            target_id=target_id,
            candidates_batch=candidates_batch,
            run_id=run_id,
            shard_id=shard_id,
        )

    def score_candidates_batch_subprocess(
        self,
        campaign_context: dict[str, Any],
        target_id: str,
        candidates_batch: List[CandidateArtifact],
        run_id: str,
        shard_id: int = 0,
    ) -> tuple[List[ObjectiveReport], dict[str, Any]]:
        """Score one shard. Returns (reports, worker_metadata)."""
        if not candidates_batch:
            return [], {"ok": True, "reports": 0, "note": "empty shard"}

        sdir = self._shard_dir(run_id, target_id, shard_id)
        in_path = sdir / "shard_input.json"
        out_path = sdir / "objective_reports.json"

        in_path.write_text(
            json.dumps(
                self._build_input(run_id, target_id, candidates_batch, campaign_context),
                indent=2,
            )
        )

        env = dict(os.environ)
        env["BINDER_ADAPTER_SBR_ROOT"] = self.config.sbr_root

        cmd = [
            self.config.python_executable,
            str(WORKER_PATH),
            "--input",
            str(in_path),
            "--output",
            str(out_path),
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.config.sbr_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise SbrSubprocessUnavailable(
                f"SBR worker timed out after {self.config.timeout_seconds}s"
            ) from exc

        if not out_path.exists():
            raise SbrSubprocessUnavailable(
                "SBR worker produced no output file.\n"
                f"exit={proc.returncode}\nstdout={proc.stdout[-2000:]}\nstderr={proc.stderr[-2000:]}"
            )

        raw = json.loads(out_path.read_text())
        reports = [ObjectiveReport.from_dict(r) for r in raw.get("reports", [])]

        meta = {
            "ok": bool(raw.get("ok")),
            "exit_code": proc.returncode,
            "capabilities": raw.get("capabilities", {}),
            "worker_errors": raw.get("worker_errors", []),
            "shard_dir": str(sdir),
            "objective_version": raw.get("objective_version"),
            "stderr_tail": proc.stderr[-800:] if proc.stderr else "",
            "backend": "subprocess",
        }
        return reports, meta
