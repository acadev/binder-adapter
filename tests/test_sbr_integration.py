"""Integration tests that execute the real StructBioReasoner scoring worker.

These are skipped automatically when the SBR checkout or its dependencies are
absent, so the suite stays green on machines without the scientific stack.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from binder_adapter.adapter_core import AdapterConfig, BinderAdapterCore

SBR_ROOT = os.environ.get("BINDER_ADAPTER_SBR_ROOT", "/Users/ramanathana/Work/StructBioReasoner")
FW = "MSTGEELQKAWDIVKRTGDKLYFRNPETGKWEWVQ"
AA = "ACDEFGHIKLMNPQRSTVWY"


def _sbr_importable() -> bool:
    if not Path(SBR_ROOT).is_dir():
        return False
    probe = (
        "import sys; sys.path.insert(0, r'%s');"
        "from struct_bio_reasoner.agents.computational_design.quality_control import SequenceQualityControl;"
        "print('ok')" % SBR_ROOT
    )
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=180)
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False


requires_sbr = pytest.mark.skipif(
    not _sbr_importable(),
    reason="StructBioReasoner (pydantic_refactor) not importable in this environment",
)


def mutate(seq: str, n: int, seed: int) -> str:
    r = random.Random(seed)
    s = list(seq)
    for p in r.sample(range(len(seq)), n):
        s[p] = r.choice([a for a in AA if a != s[p]])
    return "".join(s)


@pytest.fixture(scope="module")
def two_chain_pdbs(tmp_path_factory) -> tuple[str, str]:
    """A contacting complex and a separated one, in strict PDB column format."""
    d = tmp_path_factory.mktemp("pdbs")

    def atom(serial, name, resname, chain, resi, x, y, z, elem="C"):
        return (
            "ATOM  " f"{serial:5d}" " " f"{name:^4s}" " " f"{resname:>3s}" " "
            f"{chain:1s}" f"{resi:4d}" "    " f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{1.00:6.2f}{0.00:6.2f}" "          " f"{elem:>2s}"
        )

    def build(path: Path, y_offset: float):
        lines, serial = [], 1
        for i in range(12):
            lines.append(atom(serial, "CA", "ALA", "A", i + 1, i * 3.8, 0.0, 0.0)); serial += 1
        lines.append("TER")
        for i in range(12):
            lines.append(atom(serial, "CA", "GLY", "B", i + 1, i * 3.8, y_offset, 0.0)); serial += 1
        lines += ["TER", "END"]
        path.write_text("\n".join(lines) + "\n")

    close_pdb, far_pdb = d / "close.pdb", d / "far.pdb"
    build(close_pdb, 4.5)   # within SimpleEnergy's 5.0A cutoff
    build(far_pdb, 20.0)    # outside it
    return str(close_pdb), str(far_pdb)


def make_core(tmp_path, **kw) -> BinderAdapterCore:
    return BinderAdapterCore(
        AdapterConfig(
            sbr_root=SBR_ROOT,
            artifact_dir=str(tmp_path / "runs"),
            timeout_seconds=600,
            **kw,
        )
    )


@requires_sbr
def test_developability_comes_from_real_quality_control(tmp_path):
    """Well-formed binder should out-score a poly-A run on SBR's QC filters."""
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [
        {"hypothesis_id": "good", "candidate_sequence": mutate(FW, 10, 1)},
        {"hypothesis_id": "polyA", "candidate_sequence": "A" * len(FW)},
    ]
    res = make_core(tmp_path).score_and_select(hyps, ctx, run_id="t_dev")

    # poly-A is 100% mutated -> rejected by the budget gate before scoring
    rejected = {i["candidate_id"] for i in res["invalid_candidates"]}
    assert "polyA" in rejected

    assert "developability" in res["active_objectives"]
    ranked = {r["candidate_id"] for r in res["rankings"]}
    assert "good" in ranked

    rep = res["rankings"][0]["objective_reports"][0]
    ev = [e for e in rep["evidence"] if e["objective"] == "developability"]
    assert ev, "developability must carry provenance"
    assert "SequenceQualityControl" in ev[0]["source"]


@requires_sbr
def test_binding_affinity_is_measured_only_with_a_complex_pdb(tmp_path, two_chain_pdbs):
    close_pdb, far_pdb = two_chain_pdbs
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [
        {"hypothesis_id": "bound", "candidate_sequence": mutate(FW, 10, 2), "complex_pdb_path": close_pdb},
        {"hypothesis_id": "apart", "candidate_sequence": mutate(FW, 11, 3), "complex_pdb_path": far_pdb},
    ]
    res = make_core(tmp_path).score_and_select(hyps, ctx, run_id="t_bind")

    assert "binding_affinity" in res["active_objectives"]
    scores = {
        r["candidate_id"]: r["objective_reports"][0]["binding_affinity"]["score"]
        for r in res["rankings"]
    }
    # contacting complex must score strictly better than the separated one
    assert scores["bound"] > scores["apart"]
    assert scores["apart"] == 0.0


@requires_sbr
def test_unavailable_objectives_are_declared_never_invented(tmp_path):
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [{"hypothesis_id": "c1", "candidate_sequence": mutate(FW, 9, 4)}]
    res = make_core(tmp_path).score_and_select(hyps, ctx, run_id="t_unavail")

    unavailable = res["unavailable_objectives"]
    assert "thermostability" in unavailable
    assert "structure_confidence" in unavailable
    for name in ("thermostability", "structure_confidence"):
        assert unavailable[name], f"{name} must state why it is unavailable"
        assert name not in res["active_objectives"]

    rep = res["rankings"][0]["objective_reports"][0]
    for name in ("thermostability", "structure_confidence"):
        assert rep[name]["score"] is None, "unavailable objectives must not carry a score"
        assert rep[name]["available"] is False


@requires_sbr
def test_multi_target_produces_one_report_per_target(tmp_path):
    ctx = {"framework_sequence": FW, "target_ids": ["TA", "TB", "TC"]}
    hyps = [{"hypothesis_id": f"c{i}", "candidate_sequence": mutate(FW, 8 + i, 10 + i)} for i in range(3)]
    res = make_core(tmp_path, shard_size=2).score_and_select(hyps, ctx, run_id="t_multi")

    assert res["targets"] == ["TA", "TB", "TC"]
    for r in res["rankings"]:
        assert {rep["target_id"] for rep in r["objective_reports"]} == {"TA", "TB", "TC"}

    # 3 candidates / shard_size 2 = 2 shards per target, 3 targets = 6 shards
    assert len(res["scoring_metadata"]) == 6
    assert all(m["ok"] for m in res["scoring_metadata"])


@requires_sbr
def test_sharding_writes_auditable_artifacts(tmp_path):
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [{"hypothesis_id": f"c{i}", "candidate_sequence": mutate(FW, 8, 20 + i)} for i in range(5)]
    res = make_core(tmp_path, shard_size=2).score_and_select(hyps, ctx, run_id="t_shard")

    assert len(res["scoring_metadata"]) == 3  # ceil(5/2)
    for meta in res["scoring_metadata"]:
        shard_dir = Path(meta["shard_dir"])
        assert (shard_dir / "shard_input.json").exists()
        assert (shard_dir / "objective_reports.json").exists()
        payload = json.loads((shard_dir / "objective_reports.json").read_text())
        assert payload["ok"] is True
        assert payload["capabilities"]["quality_control"] is True


@requires_sbr
def test_all_invalid_short_circuits_to_resample(tmp_path):
    """Every candidate out of budget -> no SBR subprocess, resample requested."""
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [
        {"hypothesis_id": "too_few", "candidate_sequence": mutate(FW, 1, 30)},
        {"hypothesis_id": "too_many", "candidate_sequence": mutate(FW, 30, 31)},
    ]
    res = make_core(tmp_path).score_and_select(hyps, ctx, run_id="t_invalid")

    assert res["rankings"] == []
    assert res["needs_resample"] is True
    assert len(res["invalid_candidates"]) == 2
    assert res["scoring_metadata"] == []  # never paid for scoring
