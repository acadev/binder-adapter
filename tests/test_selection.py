from __future__ import annotations

from binder_adapter.schemas import SCHEMA_VERSION, ObjectiveReport, ObjectiveValue
from binder_adapter.selection import active_objectives, select_survivors_pareto_diverse


def report(cid: str, target: str = "T1", **scores) -> ObjectiveReport:
    """Build a report; any objective omitted from **scores is marked unavailable."""
    def val(name):
        if name in scores:
            v = scores[name]
            return ObjectiveValue.measured(v) if v is not None else ObjectiveValue.unavailable("test")
        return ObjectiveValue.unavailable("test: not supplied")

    return ObjectiveReport(
        schema_version=SCHEMA_VERSION,
        report_id=f"r:{target}:{cid}",
        candidate_id=cid,
        target_id=target,
        objective_version="test",
        binding_affinity=val("binding_affinity"),
        specificity_off_target_proxy=val("specificity_off_target_proxy"),
        developability=val("developability"),
        thermostability=val("thermostability"),
        structure_confidence=val("structure_confidence"),
    )


def test_active_objectives_requires_availability_on_every_candidate():
    by_cand = {
        "a": [report("a", developability=1.0, binding_affinity=5.0)],
        "b": [report("b", developability=0.5)],  # no binding_affinity
    }
    # developability is available for both; binding_affinity only for one -> dropped
    assert active_objectives(by_cand) == ["developability"]


def test_no_measurable_objective_withholds_ranking():
    by_cand = {"a": [report("a")], "b": [report("b")]}
    decisions, active = select_survivors_pareto_diverse(by_cand, max_survivors=5)
    assert active == []
    assert all(d.score_summary_utility is None for d in decisions)
    assert all(d.rank == 1 for d in decisions)
    assert all("withheld" in d.justification for d in decisions)


def test_dominated_candidate_lands_on_later_front():
    by_cand = {
        "best":  [report("best",  developability=1.0, binding_affinity=10.0)],
        "worse": [report("worse", developability=0.5, binding_affinity=5.0)],
    }
    decisions, active = select_survivors_pareto_diverse(by_cand, max_survivors=5)
    assert set(active) == {"developability", "binding_affinity"}
    by_id = {d.candidate_id: d for d in decisions}
    assert by_id["best"].pareto_front_id == 0
    assert by_id["worse"].pareto_front_id == 1


def test_genuine_tradeoff_shares_front_zero():
    """Neither dominates: one wins binding, the other wins developability."""
    by_cand = {
        "hi_bind": [report("hi_bind", developability=0.2, binding_affinity=10.0)],
        "hi_dev":  [report("hi_dev",  developability=1.0, binding_affinity=1.0)],
    }
    decisions, _ = select_survivors_pareto_diverse(by_cand, max_survivors=5)
    assert {d.pareto_front_id for d in decisions} == {0}


def test_multi_target_scores_are_averaged():
    by_cand = {
        "a": [
            report("a", "T1", developability=1.0),
            report("a", "T2", developability=0.0),
        ],
        "b": [
            report("b", "T1", developability=0.6),
            report("b", "T2", developability=0.6),
        ],
    }
    decisions, _ = select_survivors_pareto_diverse(by_cand, max_survivors=5)
    by_id = {d.candidate_id: d for d in decisions}
    # a averages 0.5, b averages 0.6 -> b dominates
    assert by_id["b"].pareto_front_id == 0
    assert by_id["a"].pareto_front_id == 1


def test_max_survivors_is_respected():
    by_cand = {f"c{i}": [report(f"c{i}", developability=i / 10.0)] for i in range(10)}
    decisions, _ = select_survivors_pareto_diverse(by_cand, max_survivors=3)
    assert len(decisions) == 3
    assert [d.rank for d in decisions] == [1, 2, 3]


def test_diversity_filter_skips_near_identical_sequences():
    by_cand = {
        "a": [report("a", developability=1.0)],
        "a_clone": [report("a_clone", developability=0.9)],
        "far": [report("far", developability=0.8)],
    }
    seqs = {
        "a":       "AAAAAAAAAAAAAAAAAAAA",
        "a_clone": "AAAAAAAAAAAAAAAAAAAC",  # 1/20 different
        "far":     "WWWWWWWWWWWWWWWWWWWW",  # entirely different
    }
    decisions, _ = select_survivors_pareto_diverse(
        by_cand, max_survivors=5, candidate_sequences=seqs, diversity_min_distance=0.25
    )
    picked = {d.candidate_id for d in decisions}
    assert "a" in picked
    assert "a_clone" not in picked  # too close to 'a'
    assert "far" in picked
