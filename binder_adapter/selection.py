from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import OBJECTIVE_NAMES, ObjectiveReport, RankingDecision


def active_objectives(
    objective_reports_by_candidate: dict[str, list[ObjectiveReport]],
    requested: Optional[list[str]] = None,
) -> list[str]:
    """Objectives usable for ranking: available on EVERY candidate.

    An objective that is measurable for only some candidates cannot be used for
    Pareto domination without biasing against the candidates that lack it, so we
    drop it from the comparison and report that we did.
    """
    requested = requested or list(OBJECTIVE_NAMES)
    if not objective_reports_by_candidate:
        return []

    usable = []
    for obj in requested:
        if all(
            reports and all(r.value(obj).available for r in reports)
            for reports in objective_reports_by_candidate.values()
        ):
            usable.append(obj)
    return usable


def _aggregate(reports: list[ObjectiveReport], objectives: list[str]) -> dict[str, float]:
    """Mean of each objective across the candidate's per-target reports."""
    out: dict[str, float] = {}
    for obj in objectives:
        vals = [r.value(obj).score for r in reports if r.value(obj).available]
        vals = [v for v in vals if v is not None]
        out[obj] = sum(vals) / len(vals) if vals else 0.0
    return out


def _dominates(a: dict[str, float], b: dict[str, float], objectives: list[str]) -> bool:
    """True if a is >= b on all objectives and > b on at least one (all maximize)."""
    strictly_better = False
    for obj in objectives:
        if a[obj] < b[obj]:
            return False
        if a[obj] > b[obj]:
            strictly_better = True
    return strictly_better


@dataclass(frozen=True)
class _Agg:
    candidate_id: str
    vals: dict[str, float]
    reports: list[ObjectiveReport]
    sequence: str


def _hamming_like_distance(a: str, b: str) -> float:
    if not a or not b:
        return 1.0
    n = min(len(a), len(b))
    diffs = sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))
    return diffs / float(max(len(a), len(b)))


def pareto_fronts(aggs: list[_Agg], objectives: list[str]) -> list[list[_Agg]]:
    remaining = list(aggs)
    fronts: list[list[_Agg]] = []
    while remaining:
        front = [
            r
            for r in remaining
            if not any(o is not r and _dominates(o.vals, r.vals, objectives) for o in remaining)
        ]
        if not front:  # pathological guard; avoid infinite loop
            front = list(remaining)
        fronts.append(front)
        remaining = [r for r in remaining if r not in front]
    return fronts


def select_survivors_pareto_diverse(
    objective_reports_by_candidate: dict[str, list[ObjectiveReport]],
    max_survivors: int,
    requested_objectives: Optional[list[str]] = None,
    candidate_sequences: Optional[dict[str, str]] = None,
    diversity_min_distance: float = 0.0,
) -> tuple[list[RankingDecision], list[str]]:
    """Pareto-front selection with optional sequence-diversity filtering.

    Returns (decisions, objectives_actually_used).
    """
    objectives = active_objectives(objective_reports_by_candidate, requested_objectives)
    sequences = candidate_sequences or {}

    if not objectives:
        # Nothing is comparable. Report every candidate at equal rank rather
        # than inventing an ordering.
        decisions = [
            RankingDecision(
                candidate_id=cid,
                target_id=reports[0].target_id if reports else "unknown",
                rank=1,
                pareto_front_id=0,
                diversity_group_id=0,
                score_summary_utility=None,
                justification=(
                    "no objective was measurable for all candidates; ranking withheld"
                ),
                active_objectives=[],
                objective_reports=reports,
            )
            for cid, reports in objective_reports_by_candidate.items()
        ]
        return decisions[:max_survivors], []

    aggs = [
        _Agg(
            candidate_id=cid,
            vals=_aggregate(reports, objectives),
            reports=reports,
            sequence=sequences.get(cid, ""),
        )
        for cid, reports in objective_reports_by_candidate.items()
    ]

    fronts = pareto_fronts(aggs, objectives)

    decisions: list[RankingDecision] = []
    chosen: list[_Agg] = []
    rank = 0

    for front_id, front in enumerate(fronts):
        # Within a front nothing dominates anything; order by mean utility for
        # a stable, explainable presentation order.
        front_sorted = sorted(
            front,
            key=lambda a: sum(a.vals[o] for o in objectives) / len(objectives),
            reverse=True,
        )
        for agg in front_sorted:
            if len(chosen) >= max_survivors:
                break
            if diversity_min_distance > 0.0 and agg.sequence:
                too_close = any(
                    _hamming_like_distance(agg.sequence, c.sequence) < diversity_min_distance
                    for c in chosen
                    if c.sequence
                )
                if too_close:
                    continue
            rank += 1
            chosen.append(agg)
            mean_util = sum(agg.vals[o] for o in objectives) / len(objectives)
            decisions.append(
                RankingDecision(
                    candidate_id=agg.candidate_id,
                    target_id=agg.reports[0].target_id if agg.reports else "unknown",
                    rank=rank,
                    pareto_front_id=front_id,
                    diversity_group_id=rank,
                    score_summary_utility=mean_util,
                    justification=(
                        f"Pareto front {front_id} on {len(objectives)} measurable "
                        f"objective(s): {', '.join(objectives)}"
                    ),
                    active_objectives=list(objectives),
                    objective_reports=agg.reports,
                )
            )
        if len(chosen) >= max_survivors:
            break

    return decisions, objectives
