"""OpenMM MD engine + MD-backed objectives (binding MM-GBSA, thermostability proxy).

These exercise the adapter's self-contained OpenMM pipeline — no Amber, no
molecular_simulations, no StructBioReasoner. A tiny 2-chain all-atom complex is
built in-process (via pdbfixer completing a backbone trace) so the tests need no
network and no bundled binary blobs.

Skipped automatically when OpenMM/pdbfixer are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from binder_adapter.sbr_backends import md_engine

_OK, _WHY = md_engine.openmm_available()
requires_openmm = pytest.mark.skipif(not _OK, reason=f"OpenMM/pdbfixer unavailable: {_WHY}")


@pytest.fixture(scope="module")
def synthetic_complex(tmp_path_factory) -> str:
    """A minimal 2-chain (A/B) all-atom complex, built without network access.

    We lay down a backbone trace (N, CA, C, O) for a few ALA residues per chain,
    then let pdbfixer add side chains + hydrogens so ff14SB can parameterize it.
    """
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    d = tmp_path_factory.mktemp("md_complex")
    trace = d / "trace.pdb"

    def bb(serial, name, chain, resi, x, y, z):
        elem = name[0]
        return (
            f"ATOM  {serial:5d} {name:^4s} ALA {chain}{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
        )

    lines, serial = [], 1
    for chain, yoff in (("A", 0.0), ("B", 5.0)):
        for i in range(4):
            x = i * 3.8
            for nm, dx, dy, dz in (("N", 0, 0, 0), ("CA", 0.5, 0.5, 0), ("C", 1.0, 0, 0), ("O", 1.2, 0.8, 0)):
                lines.append(bb(serial, nm, chain, i + 1, x + dx, yoff + dy, dz))
                serial += 1
        lines.append("TER")
    lines.append("END")
    trace.write_text("\n".join(lines) + "\n")

    fx = PDBFixer(filename=str(trace))
    fx.findMissingResidues()
    fx.findMissingAtoms()
    fx.addMissingAtoms()
    fx.addMissingHydrogens(7.0)
    out = d / "complex.pdb"
    with open(out, "w") as fh:
        PDBFile.writeFile(fx.topology, fx.positions, fh)
    return str(out)


def _fast_cfg_kwargs() -> dict:
    return {"equil_steps": 50, "prod_steps": 200, "sample_interval": 100}


# -- md_engine directly -------------------------------------------------------


@requires_openmm
def test_run_complex_md_produces_binding_and_stability(synthetic_complex):
    cfg = md_engine.MDConfig(**_fast_cfg_kwargs())
    res = md_engine.run_complex_md(synthetic_complex, cfg)

    b = res["binding"]
    assert isinstance(b["dg_bind_kcal_per_mol"], float)
    assert b["n_frames"] >= 1
    # dG = <E_complex> - <E_receptor> - <E_ligand> must be internally consistent
    assert b["mean_complex_energy"] == pytest.approx(
        b["dg_bind_kcal_per_mol"] + b["mean_receptor_energy"] + b["mean_ligand_energy"],
        rel=1e-4, abs=1e-2,
    )

    s = res["stability"]
    assert s["mean_rmsf_nm"] >= 0.0
    assert s["rg_std_nm"] >= 0.0
    assert "proxy" in s["note"].lower()

    p = res["provenance"]
    assert p["engine"] == "openmm"
    assert p["receptor_chains"] == ["A"]
    assert p["ligand_chains"] == ["B"]


@requires_openmm
def test_single_chain_complex_is_rejected(tmp_path):
    """A one-chain PDB has no interface -> ValueError, never a fabricated dG."""
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    trace = tmp_path / "single.pdb"
    lines, serial = [], 1
    for i in range(4):
        x = i * 3.8
        for nm, dx, dy, dz in (("N", 0, 0, 0), ("CA", 0.5, 0.5, 0), ("C", 1.0, 0, 0), ("O", 1.2, 0.8, 0)):
            lines.append(
                f"ATOM  {serial:5d} {nm:^4s} ALA A{i+1:4d}    "
                f"{x+dx:8.3f}{dy:8.3f}{dz:8.3f}  1.00  0.00          {nm[0]:>2s}"
            )
            serial += 1
    lines += ["TER", "END"]
    trace.write_text("\n".join(lines) + "\n")
    fx = PDBFixer(filename=str(trace))
    fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
    single = tmp_path / "single_fixed.pdb"
    with open(single, "w") as fh:
        PDBFile.writeFile(fx.topology, fx.positions, fh)

    with pytest.raises(ValueError, match="2 chains"):
        md_engine.run_complex_md(str(single), md_engine.MDConfig(**_fast_cfg_kwargs()))


# -- scoring.run_md_objectives wrapper ---------------------------------------


@requires_openmm
def test_md_objectives_wrapper_shapes_both_objectives(synthetic_complex):
    from binder_adapter.sbr_backends import scoring

    out = scoring.run_md_objectives(synthetic_complex, _fast_cfg_kwargs())
    assert out["binding"]["available"] is True
    assert out["thermostability"]["available"] is True
    # binding score is sign-flipped dG (higher = better)
    assert out["binding"]["score"] == pytest.approx(
        -out["binding"]["evidence"]["dg_bind_kcal_per_mol"], rel=1e-6
    )
    assert out["binding"]["evidence"]["source"] == "adapter.md_engine.openmm_mmgbsa"
    assert out["thermostability"]["evidence"]["source"] == "adapter.md_engine.trajectory_stability_proxy"


@requires_openmm
def test_md_config_accepts_receptor_ligand_chain_keys(synthetic_complex):
    """receptor_chains/ligand_chains live in md_config but are NOT MDConfig fields.

    Regression: passing them must select the interface, not raise
    'unexpected keyword argument' and silently disable MD.
    """
    from binder_adapter.sbr_backends import scoring

    cfg = dict(_fast_cfg_kwargs())
    cfg["receptor_chains"] = ["A"]
    cfg["ligand_chains"] = ["B"]
    out = scoring.run_md_objectives(synthetic_complex, cfg)
    assert out["binding"]["available"] is True, out["binding"].get("reason")
    assert out["thermostability"]["available"] is True, out["thermostability"].get("reason")
    assert out["binding"]["evidence"]["receptor_chains"] == ["A"]
    assert out["binding"]["evidence"]["ligand_chains"] == ["B"]


@requires_openmm
def test_mutation_list_built_from_sequence_diff(synthetic_complex):
    """_build_mutation_list emits correct PDBFixer 'WT-resid-MUT' strings."""
    from pdbfixer import PDBFixer

    from binder_adapter.sbr_backends import md_engine

    fixer = PDBFixer(filename=synthetic_complex)
    # chain B in the fixture is 4 ALA residues -> framework "AAAA"
    framework = "AAAA"
    candidate = "AGAV"  # positions 1->GLY, 3->VAL
    muts = md_engine._build_mutation_list(fixer, "B", framework, candidate)
    # exactly the two differing positions, as ALA-<resid>-{GLY,VAL}
    assert len(muts) == 2
    assert all(m.startswith("ALA-") for m in muts)
    assert any(m.endswith("-GLY") for m in muts)
    assert any(m.endswith("-VAL") for m in muts)


@requires_openmm
def test_threading_mutations_changes_the_simulated_structure(synthetic_complex):
    """With mutations threaded, provenance records them and atom count changes.

    ALA has fewer atoms than e.g. TRP, so substituting the binder chain's alanines
    for larger residues must change the built system — proving the mutant, not the
    wild-type, was simulated.
    """
    from binder_adapter.sbr_backends import md_engine

    cfg = md_engine.MDConfig(**_fast_cfg_kwargs())
    wt = md_engine.run_complex_md(synthetic_complex, cfg)
    mut = md_engine.run_complex_md(
        synthetic_complex, cfg,
        binder_chain="B", framework_sequence="AAAA", candidate_sequence="AWWA",
    )
    assert wt["provenance"]["n_mutations_threaded"] == 0
    assert mut["provenance"]["n_mutations_threaded"] == 2
    assert mut["provenance"]["applied_mutations"], "applied mutations must be recorded"
    # ALA->TRP adds atoms, so the mutant system is larger than the wild-type
    assert mut["provenance"]["n_atoms"] > wt["provenance"]["n_atoms"]


def test_md_objectives_unavailable_without_pdb():
    """No complex PDB -> unavailable with a reason, never a guess (no OpenMM needed)."""
    from binder_adapter.sbr_backends import scoring

    out = scoring.run_md_objectives("", {})
    assert out["binding"]["available"] is False
    assert out["thermostability"]["available"] is False
    assert "complex_pdb_path" in out["binding"]["reason"] or "not available" in out["binding"]["reason"]


def test_md_objectives_unavailable_for_missing_file():
    from binder_adapter.sbr_backends import scoring

    out = scoring.run_md_objectives("/nonexistent/complex.pdb", {})
    assert out["binding"]["available"] is False
    # message is either "does not exist" (OpenMM present) or "not available" (absent)
    assert out["binding"]["reason"]


# -- full campaign through the adapter core -----------------------------------


@requires_openmm
def test_campaign_with_md_lights_up_four_objectives(tmp_path, synthetic_complex):
    """End-to-end: md_config in the campaign upgrades binding + adds thermostability."""
    from binder_adapter.adapter_core import AdapterConfig, BinderAdapterCore

    FW = "MSTGEELQKAWDIVKRTGDKLYFRNPETGKWEWVQ"
    cand = FW[:7] + "AAA" + FW[10:]  # ~9% by hand; ensure inside 20-50%? adjust below
    # make ~30% mutations deterministically
    s = list(FW)
    for i in range(0, 33, 3):
        s[i] = "A" if s[i] != "A" else "G"
    cand = "".join(s)

    ctx = {
        "framework_sequence": FW,
        "target_ids": ["MDM2"],
        "complex_pdb_path": synthetic_complex,
        "md_config": _fast_cfg_kwargs(),
    }
    hyps = [{"hypothesis_id": "c1", "candidate_sequence": cand, "complex_pdb_path": synthetic_complex}]

    core = BinderAdapterCore(
        AdapterConfig(sbr_root="/Users/ramanathana/Work/StructBioReasoner", backend="in_process",
                      artifact_dir=str(tmp_path / "runs"))
    )
    res = core.score_and_select(hyps, ctx, run_id="md_campaign")

    # candidate must have survived the budget gate
    assert not res["invalid_candidates"], res["invalid_candidates"]
    assert "binding_affinity" in res["active_objectives"]
    assert "thermostability" in res["active_objectives"]

    rep = res["rankings"][0]["objective_reports"][0]
    assert rep["binding_affinity"]["available"] is True
    assert rep["thermostability"]["available"] is True
    ev = {e["objective"]: e for e in rep["evidence"]}
    assert ev["binding_affinity"]["source"] == "adapter.md_engine.openmm_mmgbsa"
    assert ev["thermostability"]["source"] == "adapter.md_engine.trajectory_stability_proxy"


def test_campaign_without_md_config_keeps_md_objectives_dark(tmp_path):
    """No md_config -> no MD runs; thermostability stays unavailable (no OpenMM needed)."""
    from binder_adapter.adapter_core import AdapterConfig, BinderAdapterCore

    pytest.importorskip("struct_bio_reasoner", reason="needs SBR for the static scorers")
    FW = "MSTGEELQKAWDIVKRTGDKLYFRNPETGKWEWVQ"
    s = list(FW)
    for i in range(0, 33, 3):
        s[i] = "A" if s[i] != "A" else "G"
    hyps = [{"hypothesis_id": "c1", "candidate_sequence": "".join(s)}]
    ctx = {"framework_sequence": FW, "target_ids": ["T1"]}

    core = BinderAdapterCore(
        AdapterConfig(sbr_root="/Users/ramanathana/Work/StructBioReasoner", backend="in_process")
    )
    res = core.score_and_select(hyps, ctx, run_id="no_md")
    assert "thermostability" not in res["active_objectives"]
    assert "thermostability" in res["unavailable_objectives"]
    assert "opt-in" in res["unavailable_objectives"]["thermostability"].lower()
