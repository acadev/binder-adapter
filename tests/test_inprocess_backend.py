"""Tier-2 in-process backend tests + subprocess/in-process parity.

The parity tests are the point of the Tier-2 refactor: because both backends
funnel through the shared scoring core, a report produced in-process must be
identical to one produced via subprocess for the same candidate. If these ever
diverge, the two code paths have drifted and the "single source of truth"
guarantee is broken.

Skipped automatically when the SBR checkout / deps are absent.
"""

from __future__ import annotations

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


def _sbr_importable_in_process() -> bool:
    """Tier 2 needs SBR importable in THIS interpreter, not just a subprocess."""
    if not Path(SBR_ROOT).is_dir():
        return False
    if SBR_ROOT not in sys.path:
        sys.path.insert(0, SBR_ROOT)
    try:
        from struct_bio_reasoner.agents.computational_design.quality_control import (  # noqa: F401
            SequenceQualityControl,
        )

        return True
    except Exception:
        return False


requires_sbr_inproc = pytest.mark.skipif(
    not _sbr_importable_in_process(),
    reason="StructBioReasoner (pydantic_refactor) not importable in-process",
)


def mutate(seq: str, n: int, seed: int) -> str:
    r = random.Random(seed)
    s = list(seq)
    for p in r.sample(range(len(seq)), n):
        s[p] = r.choice([a for a in AA if a != s[p]])
    return "".join(s)


def make_core(tmp_path, backend: str, **kw) -> BinderAdapterCore:
    return BinderAdapterCore(
        AdapterConfig(
            sbr_root=SBR_ROOT,
            artifact_dir=str(tmp_path / f"runs_{backend}"),
            timeout_seconds=600,
            backend=backend,
            **kw,
        )
    )


@pytest.fixture(scope="module")
def two_chain_pdbs(tmp_path_factory) -> tuple[str, str]:
    d = tmp_path_factory.mktemp("pdbs_inproc")

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
    build(close_pdb, 4.5)
    build(far_pdb, 20.0)
    return str(close_pdb), str(far_pdb)


# -- Tier-2 correctness -------------------------------------------------------


@requires_sbr_inproc
def test_in_process_backend_measures_developability(tmp_path):
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [{"hypothesis_id": "good", "candidate_sequence": mutate(FW, 10, 1)}]
    res = make_core(tmp_path, "in_process").score_and_select(hyps, ctx, run_id="t_inproc_dev")

    assert "developability" in res["active_objectives"]
    rep = res["rankings"][0]["objective_reports"][0]
    ev = [e for e in rep["evidence"] if e["objective"] == "developability"]
    assert ev and "SequenceQualityControl" in ev[0]["source"]
    # The metadata must announce this ran in-process, not via subprocess.
    assert all(m.get("backend") == "in_process" for m in res["scoring_metadata"])


@requires_sbr_inproc
def test_in_process_backend_writes_same_artifacts(tmp_path):
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [{"hypothesis_id": f"c{i}", "candidate_sequence": mutate(FW, 8, 20 + i)} for i in range(5)]
    res = make_core(tmp_path, "in_process", shard_size=2).score_and_select(
        hyps, ctx, run_id="t_inproc_shard"
    )
    assert len(res["scoring_metadata"]) == 3  # ceil(5/2)
    for meta in res["scoring_metadata"]:
        shard_dir = Path(meta["shard_dir"])
        assert (shard_dir / "shard_input.json").exists()
        assert (shard_dir / "objective_reports.json").exists()


@requires_sbr_inproc
def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        BinderAdapterCore(AdapterConfig(sbr_root=SBR_ROOT, backend="nonsense"))


# -- Parity: the whole reason the scoring core is shared ----------------------


def _strip_volatile(report: dict) -> dict:
    """Drop fields that legitimately differ run-to-run (paths, etc.).

    Objective scores, availability, reasons and evidence sources must match; the
    absolute PDB path inside binding-affinity evidence is environment noise.
    """
    r = dict(report)
    for ev in r.get("evidence", []):
        ev.pop("pdb", None)
    return r


@requires_sbr_inproc
def test_subprocess_and_in_process_reports_are_identical(tmp_path, two_chain_pdbs):
    close_pdb, _ = two_chain_pdbs
    ctx = {"framework_sequence": FW, "target_ids": ["TA", "TB"]}
    hyps = [
        {"hypothesis_id": "a", "candidate_sequence": mutate(FW, 10, 2), "complex_pdb_path": close_pdb},
        {"hypothesis_id": "b", "candidate_sequence": mutate(FW, 12, 3)},
        {"hypothesis_id": "c", "candidate_sequence": mutate(FW, 9, 4)},
    ]

    sub = make_core(tmp_path, "subprocess").score_and_select(hyps, dict(ctx), run_id="parity")
    ip = make_core(tmp_path, "in_process").score_and_select(hyps, dict(ctx), run_id="parity")

    def reports_by_key(res):
        out = {}
        for ranking in res["rankings"]:
            for rep in ranking["objective_reports"]:
                out[(rep["candidate_id"], rep["target_id"])] = _strip_volatile(rep)
        return out

    sub_reports = reports_by_key(sub)
    ip_reports = reports_by_key(ip)

    assert set(sub_reports) == set(ip_reports)
    for key in sub_reports:
        for obj in (
            "binding_affinity",
            "specificity_off_target_proxy",
            "developability",
            "thermostability",
            "structure_confidence",
        ):
            assert sub_reports[key][obj] == ip_reports[key][obj], f"{obj} differs for {key}"

    # Selection outcomes must agree too.
    assert sub["active_objectives"] == ip["active_objectives"]
    assert [r["candidate_id"] for r in sub["rankings"]] == [
        r["candidate_id"] for r in ip["rankings"]
    ]


@requires_sbr_inproc
def test_both_backends_declare_thermostability_unavailable(tmp_path):
    """Thermostability stays honestly unavailable in BOTH tiers (mmpbsa deps absent)."""
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}
    hyps = [{"hypothesis_id": "c1", "candidate_sequence": mutate(FW, 9, 7)}]
    for backend in ("subprocess", "in_process"):
        res = make_core(tmp_path, backend).score_and_select(hyps, ctx, run_id=f"t_thermo_{backend}")
        unavailable = res["unavailable_objectives"]
        assert "thermostability" in unavailable
        assert "molecular_simulations" in unavailable["thermostability"]
        assert "thermostability" not in res["active_objectives"]
