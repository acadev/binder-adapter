"""Self-contained OpenMM MD engine for physics-based objectives.

This is the adapter's own implementation of the build -> minimize -> equilibrate
-> production -> free-energy pipeline. It deliberately does NOT use
StructBioReasoner's molecular_dynamics package: those modules (MD.py,
distributed.py, mmpbsa_agent.py) are Parsl orchestration shells whose actual
physics lives in the external `molecular_simulations` package, which shells out
to a full Amber install. None of that is required here.

Everything below runs on OpenMM alone:
  * force field: bundled amber14 ff14SB protein parameters (no Amber binaries)
  * solvent:     bundled GB implicit-solvent models (gbn2 by default)
  * prep:        pdbfixer (adds missing atoms/H, clean against numpy 1.26 / openmm 8.3)

Two objectives come out of one short implicit-solvent trajectory:

  MM-GBSA binding free energy (single-trajectory approximation)
     dG_bind = <E_complex> - <E_receptor> - <E_ligand>, averaged over frames of
     the complex trajectory (receptor/ligand energies computed from the same
     frames, split by chain). Negative = favourable. This is a real physics-based
     dG, upgrading the crude contact-count binding proxy.

  Conformational-stability proxy for thermostability
     From the same trajectory: backbone RMSF, radius-of-gyration drift, and
     potential-energy variance. Tighter dynamics => more stable fold. This is an
     explicitly-labelled PROXY, not a true folding ddG/dTm (which would need
     alchemical FEP), and it is reported as such.

Fidelity is configurable via MDConfig. Defaults are DEMO scale (picoseconds, CPU)
so the pipeline is provable end-to-end quickly; bump steps + use a GPU platform
for production runs.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# OpenMM / pdbfixer are optional at import time so the adapter still loads (and
# reports the objective unavailable) in an environment without them.
_IMPORT_ERROR = ""
try:
    import numpy as np
    from openmm import LangevinMiddleIntegrator, Platform, VerletIntegrator, unit
    from openmm import app
    from openmm.app import Modeller, PDBFile
    from pdbfixer import PDBFixer

    _OPENMM_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    _OPENMM_AVAILABLE = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def openmm_available() -> tuple[bool, str]:
    """(available, reason_if_not). Cheap to call; used for objective gating."""
    return _OPENMM_AVAILABLE, _IMPORT_ERROR


@dataclass(frozen=True)
class MDConfig:
    """MD run parameters.

    Defaults are DEMO scale: fast, CPU-friendly, enough to prove the pipeline and
    give a meaningful (if noisy) estimate. For production, raise
    equil_steps/prod_steps into the 10^5-10^6 range and set platform="CUDA".
    """

    force_field: str = "amber14/protein.ff14SB.xml"
    implicit_solvent: str = "implicit/gbn2.xml"
    temperature_kelvin: float = 300.0
    friction_per_ps: float = 1.0
    timestep_ps: float = 0.002
    minimize_iterations: int = 200
    equil_steps: int = 500          # ~1 ps demo
    prod_steps: int = 2000          # ~4 ps demo
    sample_interval: int = 200      # frames every 0.4 ps -> ~10 frames
    nonbonded_cutoff_nm: float = 2.0
    platform: str = "CPU"           # "CPU" | "OpenCL" | "CUDA" | "Reference"
    ph: float = 7.0
    add_missing_hydrogens: bool = True


class MDUnavailable(RuntimeError):
    """Raised when OpenMM/pdbfixer are not importable."""


def _require_openmm() -> None:
    if not _OPENMM_AVAILABLE:
        raise MDUnavailable(f"OpenMM/pdbfixer not available: {_IMPORT_ERROR}")


def _make_forcefield(cfg: MDConfig):
    return app.ForceField(cfg.force_field, cfg.implicit_solvent)


_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
_THREE_TO_ONE = {v: k for k, v in _ONE_TO_THREE.items()}


def _chain_resnames(fixer, chain_id: str) -> list[tuple[int, str]]:
    """(residue.id, resname) for standard AA residues in a chain, in order."""
    out = []
    for chain in fixer.topology.chains():
        if chain.id != chain_id:
            continue
        for res in chain.residues():
            if res.name.upper() in _THREE_TO_ONE:
                out.append((res.id, res.name.upper()))
    return out


def _build_mutation_list(fixer, chain_id: str, framework: str, candidate: str) -> list[str]:
    """PDBFixer mutation strings ('WT-<resid>-MUT') for positions that differ.

    Aligns the candidate to the chain's actual residues positionally. Only
    substitutions are threaded (the mutation budget already forbids indels for
    binder design here). Positions whose WT one-letter doesn't match the
    framework are skipped with no mutation (keeps geometry safe).
    """
    residues = _chain_resnames(fixer, chain_id)
    muts: list[str] = []
    n = min(len(residues), len(framework), len(candidate))
    for i in range(n):
        resid, wt_three = residues[i]
        cand_aa = candidate[i].upper()
        if cand_aa == framework[i].upper():
            continue
        if cand_aa not in _ONE_TO_THREE:
            continue
        # sanity: structure WT should match framework WT at this position
        if _THREE_TO_ONE.get(wt_three) != framework[i].upper():
            continue
        muts.append(f"{wt_three}-{resid}-{_ONE_TO_THREE[cand_aa]}")
    return muts


def _prepare_structure(
    pdb_path: str,
    cfg: MDConfig,
    mutate_chain: Optional[str] = None,
    framework: Optional[str] = None,
    candidate: Optional[str] = None,
):
    """Load + repair a PDB with pdbfixer. Returns (topology, positions).

    When ``mutate_chain`` + ``framework`` + ``candidate`` are given, the
    candidate's substitutions (relative to the framework) are threaded onto that
    chain via PDBFixer.applyMutations BEFORE atoms/H are added, so each candidate
    is simulated as its own mutant structure rather than the wild-type.
    """
    fixer = PDBFixer(filename=str(pdb_path))

    applied_mutations: list[str] = []
    if mutate_chain and framework and candidate:
        applied_mutations = _build_mutation_list(fixer, mutate_chain, framework, candidate)
        if applied_mutations:
            try:
                fixer.applyMutations(applied_mutations, mutate_chain)
            except (KeyError, ValueError) as exc:
                # A mutation PDBFixer can't place (unknown template etc.) must not
                # silently simulate the wild-type; signal so the caller records it.
                raise ValueError(
                    f"applyMutations failed for chain {mutate_chain}: {exc}"
                ) from exc

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    if cfg.add_missing_hydrogens:
        fixer.addMissingHydrogens(cfg.ph)
    return fixer.topology, fixer.positions, applied_mutations


def _potential_energy(topology, positions, ff, cfg: MDConfig) -> float:
    """Single-point potential energy (kcal/mol) for a topology+positions."""
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=cfg.nonbonded_cutoff_nm * unit.nanometer,
        constraints=None,
    )
    integ = VerletIntegrator(0.001 * unit.picoseconds)
    plat = Platform.getPlatformByName(cfg.platform)
    ctx = app.Simulation(topology, system, integ, plat).context
    ctx.setPositions(positions)
    return ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilocalorie_per_mole
    )


def _subset_by_chain(topology, positions, keep_chain_ids: set[str]):
    m = Modeller(topology, positions)
    to_delete = [c for c in m.topology.chains() if c.id not in keep_chain_ids]
    m.delete(to_delete)
    return m.topology, m.positions


def _chain_ids(topology) -> list[str]:
    return [c.id for c in topology.chains()]


def run_complex_md(
    complex_pdb_path: str,
    cfg: Optional[MDConfig] = None,
    receptor_chains: Optional[list[str]] = None,
    ligand_chains: Optional[list[str]] = None,
    binder_chain: Optional[str] = None,
    framework_sequence: Optional[str] = None,
    candidate_sequence: Optional[str] = None,
) -> dict[str, Any]:
    """Build, minimize, equilibrate and run production MD on a complex PDB.

    Splits chains into receptor vs. ligand (defaults: first chain = receptor,
    the rest = ligand), then computes single-trajectory MM-GBSA dG_bind and a
    conformational-stability proxy from the complex trajectory.

    When ``binder_chain`` + ``framework_sequence`` + ``candidate_sequence`` are
    given, the candidate's substitutions are threaded onto the binder chain so
    the simulation reflects the actual mutant, not the wild-type template. This
    is what makes per-candidate scores mean something.

    Returns a result dict with both objective payloads plus provenance. Raises
    MDUnavailable if OpenMM/pdbfixer are missing; raises ValueError for a
    single-chain structure (no interface to score).
    """
    _require_openmm()
    cfg = cfg or MDConfig()

    top, pos, applied_mutations = _prepare_structure(
        complex_pdb_path,
        cfg,
        mutate_chain=binder_chain,
        framework=framework_sequence,
        candidate=candidate_sequence,
    )
    all_chains = _chain_ids(top)
    if len(all_chains) < 2:
        raise ValueError(
            f"complex must have >=2 chains for binding free energy; found {all_chains}"
        )

    rec = set(receptor_chains) if receptor_chains else {all_chains[0]}
    lig = set(ligand_chains) if ligand_chains else set(all_chains) - rec
    if not lig:
        raise ValueError("ligand chain set is empty after receptor assignment")

    ff = _make_forcefield(cfg)

    # --- build complex system + simulation ---
    system = ff.createSystem(
        top,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=cfg.nonbonded_cutoff_nm * unit.nanometer,
        constraints=app.HBonds,
    )
    integ = LangevinMiddleIntegrator(
        cfg.temperature_kelvin * unit.kelvin,
        cfg.friction_per_ps / unit.picosecond,
        cfg.timestep_ps * unit.picoseconds,
    )
    plat = Platform.getPlatformByName(cfg.platform)
    sim = app.Simulation(top, system, integ, plat)
    sim.context.setPositions(pos)

    # --- minimize + equilibrate ---
    # Thorough minimization (tolerance-based) matters more than a fixed iteration
    # cap for mutant structures with clashy repacked side chains — it's the main
    # guard against downstream 'coordinate is NaN' blow-ups.
    sim.minimizeEnergy(
        tolerance=10.0 * unit.kilojoule_per_mole / unit.nanometer,
        maxIterations=max(cfg.minimize_iterations, 500),
    )

    def _equilibrate_safely() -> bool:
        """Equilibrate in small chunks; return False if the run went non-finite."""
        sim.context.setVelocitiesToTemperature(cfg.temperature_kelvin * unit.kelvin)
        if cfg.equil_steps <= 0:
            return True
        chunk = max(1, min(50, cfg.equil_steps))
        done = 0
        while done < cfg.equil_steps:
            sim.step(min(chunk, cfg.equil_steps - done))
            done += chunk
            e = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
                unit.kilocalorie_per_mole
            )
            if not np.isfinite(e):
                return False
        return True

    # A clashy mutant can blow up during equilibration (before any production
    # frame). Retry once with a harder re-minimization + fresh velocities rather
    # than losing the candidate to a transient instability.
    if not _equilibrate_safely():
        sim.context.setPositions(pos)
        sim.minimizeEnergy(
            tolerance=5.0 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max(cfg.minimize_iterations, 2000),
        )
        if not _equilibrate_safely():
            raise ValueError(
                "structure remained unstable through two equilibration attempts "
                "(NaN energy); no score produced"
            )

    # --- production with sampling ---
    complex_e, receptor_e, ligand_e = [], [], []
    rg_series, positions_series = [], []
    skipped_frames = 0
    n_samples = max(1, cfg.prod_steps // max(1, cfg.sample_interval))

    for _ in range(n_samples):
        sim.step(cfg.sample_interval)
        state = sim.context.getState(getEnergy=True, getPositions=True)
        frame_pos = state.getPositions(asNumpy=True)

        e_c = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        # A blown-up frame (NaN/inf energy or coords) is dropped, not fatal —
        # one unstable frame shouldn't discard the candidate's whole trajectory.
        xyz = frame_pos.value_in_unit(unit.nanometer)
        if not np.isfinite(e_c) or not np.isfinite(xyz).all():
            skipped_frames += 1
            continue

        # receptor / ligand single-point energies from the SAME frame (split by chain)
        t_r, p_r = _subset_by_chain(top, frame_pos, rec)
        t_l, p_l = _subset_by_chain(top, frame_pos, lig)
        e_r = _potential_energy(t_r, p_r, ff, cfg)
        e_l = _potential_energy(t_l, p_l, ff, cfg)
        if not (np.isfinite(e_r) and np.isfinite(e_l)):
            skipped_frames += 1
            continue

        complex_e.append(e_c)
        receptor_e.append(e_r)
        ligand_e.append(e_l)
        rg_series.append(_radius_of_gyration(xyz))
        positions_series.append(xyz)

    if not complex_e:
        raise ValueError(
            f"all {n_samples} MD frames were non-finite (unstable structure); no score produced"
        )

    binding = _mmgbsa_from_series(complex_e, receptor_e, ligand_e)
    stability = _stability_proxy(complex_e, rg_series, positions_series)

    provenance = {
        "engine": "openmm",
        "openmm_version": _openmm_version(),
        "force_field": cfg.force_field,
        "implicit_solvent": cfg.implicit_solvent,
        "platform": cfg.platform,
        "receptor_chains": sorted(rec),
        "ligand_chains": sorted(lig),
        "n_frames": len(complex_e),
        "prod_steps": cfg.prod_steps,
        "equil_steps": cfg.equil_steps,
        "temperature_kelvin": cfg.temperature_kelvin,
        "n_atoms": top.getNumAtoms(),
        "demo_scale": cfg.prod_steps < 50000,
        "binder_chain": binder_chain or "",
        "applied_mutations": applied_mutations,
        "n_mutations_threaded": len(applied_mutations),
        "skipped_frames": skipped_frames,
    }
    return {"binding": binding, "stability": stability, "provenance": provenance}


def _radius_of_gyration(xyz) -> float:
    """Rg in nm for an (N,3) coordinate array (unweighted)."""
    center = xyz.mean(axis=0)
    d2 = ((xyz - center) ** 2).sum(axis=1)
    return float(np.sqrt(d2.mean()))


def _mmgbsa_from_series(
    complex_e: list[float], receptor_e: list[float], ligand_e: list[float]
) -> dict[str, Any]:
    """Single-trajectory MM-GBSA: dG = <E_c> - <E_r> - <E_l> (kcal/mol)."""
    dg_frames = [c - r - l for c, r, l in zip(complex_e, receptor_e, ligand_e)]
    mean_dg = statistics.fmean(dg_frames)
    std_dg = statistics.pstdev(dg_frames) if len(dg_frames) > 1 else 0.0
    return {
        "dg_bind_kcal_per_mol": mean_dg,
        "dg_bind_std": std_dg,
        "mean_complex_energy": statistics.fmean(complex_e),
        "mean_receptor_energy": statistics.fmean(receptor_e),
        "mean_ligand_energy": statistics.fmean(ligand_e),
        "n_frames": len(dg_frames),
    }


def _stability_proxy(
    complex_e: list[float], rg_series: list[float], positions_series: list
) -> dict[str, Any]:
    """Conformational-stability proxy from the complex trajectory.

    Combines three signals (lower = more stable):
      - potential-energy std over frames
      - radius-of-gyration std (compactness drift)
      - mean per-atom coordinate RMSF vs. the trajectory-average structure
    """
    e_std = statistics.pstdev(complex_e) if len(complex_e) > 1 else 0.0
    rg_std = statistics.pstdev(rg_series) if len(rg_series) > 1 else 0.0

    rmsf = 0.0
    if len(positions_series) > 1:
        arr = np.stack(positions_series, axis=0)  # (frames, atoms, 3)
        mean_struct = arr.mean(axis=0)
        per_atom = np.sqrt(((arr - mean_struct) ** 2).sum(axis=2).mean(axis=0))
        rmsf = float(per_atom.mean())  # nm

    return {
        "potential_energy_std": e_std,
        "rg_std_nm": rg_std,
        "mean_rmsf_nm": rmsf,
        "n_frames": len(complex_e),
        "note": (
            "conformational-stability proxy (implicit-solvent MD); NOT a folding "
            "ddG/dTm, which requires alchemical FEP"
        ),
    }


def _openmm_version() -> str:
    try:
        import openmm

        return openmm.__version__
    except Exception:  # pragma: no cover
        return "unknown"
