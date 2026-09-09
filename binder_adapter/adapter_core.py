from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from .constraints import mutation_count_bounds, validate_mutation_constraints
from .extraction import extract_candidate_artifacts_from_jnana
from .sbr_backends.inprocess_backend import (
    InProcessBackendConfig,
    InProcessScoringBackend,
)
from .sbr_backends.subprocess_backend import (
    SbrSubprocessUnavailable,
    SubprocessBackendConfig,
    SubprocessScoringBackend,
)
from .schemas import (
    OBJECTIVE_NAMES,
    SCHEMA_VERSION,
    CandidateArtifact,
    InvalidCandidate,
    ObjectiveReport,
)
from .selection import select_survivors_pareto_diverse


def shard(items: list, size: int) -> list[list]:
    if size <= 0:
        return [list(items)]
    return [items[i : i + size] for i in range(0, len(items), size)]


@dataclass
class AdapterConfig:
    sbr_root: str
    python_executable: str = sys.executable
    min_mutated_fraction: float = 0.20
    max_mutated_fraction: float = 0.50
    shard_size: int = 16
    max_survivors: int = 10
    diversity_min_distance: float = 0.0
    artifact_dir: Optional[str] = None
    objectives: list[str] = field(default_factory=lambda: list(OBJECTIVE_NAMES))
    specificity_reference_sequences: list[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    # "subprocess" (Tier 1, default — process isolation per shard) or
    # "in_process" (Tier 2 — direct SBR calls in the host interpreter).
    backend: str = "subprocess"


class BinderAdapterCore:
    """Shared core behind both front-ends (adapter CLI and Jnana ranking hook)."""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.backend = self._build_backend(config)

    @staticmethod
    def _build_backend(config: AdapterConfig):
        if config.backend == "in_process":
            return InProcessScoringBackend(
                InProcessBackendConfig(
                    sbr_root=config.sbr_root,
                    artifact_dir=config.artifact_dir,
                    specificity_reference_sequences=list(
                        config.specificity_reference_sequences
                    ),
                )
            )
        if config.backend == "subprocess":
            return SubprocessScoringBackend(
                SubprocessBackendConfig(
                    sbr_root=config.sbr_root,
                    python_executable=config.python_executable,
                    timeout_seconds=config.timeout_seconds,
                    artifact_dir=config.artifact_dir,
                    specificity_reference_sequences=list(
                        config.specificity_reference_sequences
                    ),
                )
            )
        raise ValueError(
            f"unknown backend {config.backend!r}; expected 'subprocess' or 'in_process'"
        )

    # -- gate -------------------------------------------------------------

    def validate(
        self, candidates: list[CandidateArtifact]
    ) -> tuple[list[CandidateArtifact], list[InvalidCandidate]]:
        valid: list[CandidateArtifact] = []
        invalid: list[InvalidCandidate] = []
        for c in candidates:
            res = validate_mutation_constraints(
                candidate=c,
                framework_sequence=c.framework_sequence,
                min_mutated_fraction=self.config.min_mutated_fraction,
                max_mutated_fraction=self.config.max_mutated_fraction,
            )
            if res.valid:
                valid.append(c)
            elif res.invalid is not None:
                invalid.append(res.invalid)
        return valid, invalid

    # -- score ------------------------------------------------------------

    def score(
        self,
        candidates: list[CandidateArtifact],
        campaign_context: dict[str, Any],
        target_id: str,
        run_id: str,
    ) -> tuple[list[ObjectiveReport], list[dict[str, Any]]]:
        reports: list[ObjectiveReport] = []
        metas: list[dict[str, Any]] = []
        for shard_id, batch in enumerate(shard(candidates, self.config.shard_size)):
            batch_reports, meta = self.backend.score_candidates_batch(
                campaign_context=campaign_context,
                target_id=target_id,
                candidates_batch=batch,
                run_id=run_id,
                shard_id=shard_id,
            )
            reports.extend(batch_reports)
            metas.append({"shard_id": shard_id, "candidates": len(batch), **meta})
        return reports, metas

    # -- full round -------------------------------------------------------

    def score_and_select(
        self,
        hypotheses: list[dict[str, Any]],
        campaign_context: dict[str, Any],
        run_id: str,
        target_ids: Optional[list[str]] = None,
        max_survivors: Optional[int] = None,
    ) -> dict[str, Any]:
        """Extract -> gate on mutation budget -> score via SBR -> Pareto select.

        Multi-target: every valid candidate is scored against every target, and
        selection aggregates across targets (mean per objective).
        """
        max_survivors = max_survivors or self.config.max_survivors

        targets = target_ids or campaign_context.get("target_ids") or campaign_context.get("targets")
        if isinstance(targets, str):
            targets = [targets]
        if not targets:
            raise ValueError("at least one target_id is required")
        targets = [str(t) for t in targets]

        candidates = extract_candidate_artifacts_from_jnana(hypotheses, campaign_context)
        valid, invalid = self.validate(candidates)

        L = len(str(campaign_context["framework_sequence"]))
        min_count, max_count = mutation_count_bounds(
            L, self.config.min_mutated_fraction, self.config.max_mutated_fraction
        )
        budget = {
            "framework_length": L,
            "min_mutated_fraction": self.config.min_mutated_fraction,
            "max_mutated_fraction": self.config.max_mutated_fraction,
            "min_mutated_count": min_count,
            "max_mutated_count": max_count,
        }

        if not valid:
            return {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "targets": targets,
                "mutation_budget": budget,
                "rankings": [],
                "invalid_candidates": [i.to_dict() for i in invalid],
                "needs_resample": True,
                "resample_shortfall": max_survivors,
                "active_objectives": [],
                "unavailable_objectives": {},
                "scoring_metadata": [],
            }

        by_candidate: dict[str, list[ObjectiveReport]] = {}
        metas: list[dict[str, Any]] = []
        scoring_error: Optional[str] = None

        for target_id in targets:
            try:
                reports, target_metas = self.score(valid, campaign_context, target_id, run_id)
            except SbrSubprocessUnavailable as exc:
                scoring_error = str(exc)
                break
            metas.extend(target_metas)
            for r in reports:
                by_candidate.setdefault(r.candidate_id, []).append(r)

        if scoring_error is not None:
            return {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "targets": targets,
                "mutation_budget": budget,
                "rankings": [],
                "invalid_candidates": [i.to_dict() for i in invalid],
                "needs_resample": False,
                "scoring_error": scoring_error,
                "active_objectives": [],
                "unavailable_objectives": {},
                "scoring_metadata": metas,
            }

        sequences = {c.candidate_id: c.candidate_sequence for c in valid}
        decisions, active = select_survivors_pareto_diverse(
            objective_reports_by_candidate=by_candidate,
            max_survivors=max_survivors,
            requested_objectives=self.config.objectives,
            candidate_sequences=sequences,
            diversity_min_distance=self.config.diversity_min_distance,
        )

        unavailable: dict[str, str] = {}
        for reports in by_candidate.values():
            for r in reports:
                for obj in OBJECTIVE_NAMES:
                    v = r.value(obj)
                    if not v.available and obj not in unavailable:
                        unavailable[obj] = v.unavailable_reason

        shortfall = max(0, max_survivors - len(decisions))
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "targets": targets,
            "mutation_budget": budget,
            "rankings": [d.to_dict() for d in decisions],
            "invalid_candidates": [i.to_dict() for i in invalid],
            "needs_resample": bool(invalid) or shortfall > 0,
            "resample_shortfall": shortfall,
            "active_objectives": active,
            "unavailable_objectives": unavailable,
            "scoring_metadata": metas,
        }


def build_adapter_core(sbr_root: str, **kwargs: Any) -> BinderAdapterCore:
    return BinderAdapterCore(AdapterConfig(sbr_root=sbr_root, **kwargs))
