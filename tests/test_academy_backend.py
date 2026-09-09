"""Academy/Parsl backend tests + parity with the in-process backend.

The whole point of the scale-out backend is that it must NOT change the science:
a report produced by fanning a shard out through Parsl must be byte-identical to
one produced by the in-process backend for the same candidate. If these diverge,
the "single source of truth" scoring core has been bypassed.

Skipped automatically when academy/parsl or the SBR checkout are absent.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

from binder_adapter.adapter_core import AdapterConfig, BinderAdapterCore

SBR_ROOT = os.environ.get("BINDER_ADAPTER_SBR_ROOT", "/Users/ramanathana/Work/StructBioReasoner")
FW = "MSTGEELQKAWDIVKRTGDKLYFRNPETGKWEWVQ"
AA = "ACDEFGHIKLMNPQRSTVWY"


def _deps_ok() -> bool:
    if not Path(SBR_ROOT).is_dir():
        return False
    if SBR_ROOT not in sys.path:
        sys.path.insert(0, SBR_ROOT)
    try:
        import academy  # noqa: F401
        import parsl  # noqa: F401
        from struct_bio_reasoner.agents.computational_design.quality_control import (  # noqa: F401
            SequenceQualityControl,
        )

        return True
    except Exception:
        return False


requires_deps = pytest.mark.skipif(
    not _deps_ok(), reason="academy/parsl or StructBioReasoner not importable"
)


def mutate(seq: str, n: int, seed: int) -> str:
    r = random.Random(seed)
    s = list(seq)
    for p in r.sample(range(len(seq)), n):
        s[p] = r.choice([a for a in AA if a != s[p]])
    return "".join(s)


def _strip_volatile(report: dict) -> dict:
    r = dict(report)
    for ev in r.get("evidence", []):
        ev.pop("pdb", None)
    return r


def _make_academy_core(tmp_path) -> BinderAdapterCore:
    """A core whose scoring backend is the Academy/Parsl one (local provider)."""
    from binder_adapter.sbr_backends.academy_backend import (
        AcademyBackendConfig,
        AcademyScoringBackend,
        default_local_parsl_config,
    )

    core = BinderAdapterCore(
        AdapterConfig(
            sbr_root=SBR_ROOT,
            backend="in_process",
            artifact_dir=str(tmp_path / "runs_academy"),
            shard_size=2,
        )
    )
    core.backend = AcademyScoringBackend(
        AcademyBackendConfig(
            sbr_root=SBR_ROOT,
            parsl_config_factory=lambda: default_local_parsl_config(2),
            artifact_dir=str(tmp_path / "runs_academy"),
        )
    )
    return core


@requires_deps
def test_academy_backend_available():
    from binder_adapter.sbr_backends.academy_backend import academy_available

    ok, why = academy_available()
    assert ok, why


@requires_deps
def test_academy_backend_scores_and_shards(tmp_path):
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [{"hypothesis_id": f"c{i}", "candidate_sequence": mutate(FW, 8, 30 + i)} for i in range(5)]
    core = _make_academy_core(tmp_path)
    try:
        res = core.score_and_select(hyps, ctx, run_id="t_academy_shard")
    finally:
        core.backend.shutdown()

    assert "developability" in res["active_objectives"]
    assert len(res["scoring_metadata"]) == 3  # ceil(5/2) shards
    assert all(m.get("backend") == "academy_parsl" for m in res["scoring_metadata"])
    for meta in res["scoring_metadata"]:
        d = Path(meta["shard_dir"])
        assert (d / "shard_input.json").exists()
        assert (d / "objective_reports.json").exists()


@requires_deps
def test_academy_and_in_process_reports_are_identical(tmp_path):
    """Parity: Parsl fan-out must produce identical objective reports."""
    ctx = {"framework_sequence": FW, "target_ids": ["TA", "TB"]}
    hyps = [
        {"hypothesis_id": "a", "candidate_sequence": mutate(FW, 10, 2)},
        {"hypothesis_id": "b", "candidate_sequence": mutate(FW, 12, 3)},
        {"hypothesis_id": "c", "candidate_sequence": mutate(FW, 9, 4)},
    ]

    ip_core = BinderAdapterCore(
        AdapterConfig(
            sbr_root=SBR_ROOT, backend="in_process",
            artifact_dir=str(tmp_path / "runs_ip"), shard_size=2,
        )
    )
    ip = ip_core.score_and_select(hyps, dict(ctx), run_id="parity")

    ac_core = _make_academy_core(tmp_path)
    try:
        ac = ac_core.score_and_select(hyps, dict(ctx), run_id="parity")
    finally:
        ac_core.backend.shutdown()

    def reports_by_key(res):
        out = {}
        for ranking in res["rankings"]:
            for rep in ranking["objective_reports"]:
                out[(rep["candidate_id"], rep["target_id"])] = _strip_volatile(rep)
        return out

    ip_r, ac_r = reports_by_key(ip), reports_by_key(ac)
    assert set(ip_r) == set(ac_r)
    for key in ip_r:
        for obj in (
            "binding_affinity",
            "specificity_off_target_proxy",
            "developability",
            "thermostability",
            "structure_confidence",
        ):
            assert ip_r[key][obj] == ac_r[key][obj], f"{obj} differs for {key}"

    assert ip["active_objectives"] == ac["active_objectives"]
    assert [r["candidate_id"] for r in ip["rankings"]] == [
        r["candidate_id"] for r in ac["rankings"]
    ]
