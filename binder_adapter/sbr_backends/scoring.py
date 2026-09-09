"""Shared StructBioReasoner scoring core — the single source of truth.

Both scoring backends call these functions:

  * ``sbr_scoring_worker.py`` runs them in a subprocess (Tier 1). The subprocess
    isolates the SBR import graph (MDAnalysis, academy, parsl, ...) so a heavy or
    crashy dependency cannot take down the host process, and so each shard is an
    independent unit of work for a Slurm/PBS array.
  * ``InProcessScoringBackend`` calls them directly in the host interpreter
    (Tier 2). Faster — no per-shard process spawn or JSON round-trip — for when
    the caller already lives in an environment that can import SBR.

Because both paths funnel through this module, a subprocess report and an
in-process report for the same candidate are byte-for-byte identical. There is
no second copy of the scoring rules to drift out of sync.

Design rule (unchanged from Tier 1): an objective is either MEASURED from real
SBR code, or it is marked unavailable with a machine-readable reason. Nothing is
ever guessed or back-filled.

Objective sources on the ``pydantic_refactor`` branch:
  developability                -> computational_design.quality_control.SequenceQualityControl (8 filters)
  binding_affinity              -> computational_design.energy.SimpleEnergy (needs a complex PDB)
  specificity_off_target_proxy  -> sequence-novelty proxy vs. a caller-supplied panel
                                   (embedding route needs genslm_esm, not installed)
  thermostability               -> UNAVAILABLE: MM-PBSA compute path imports
                                   `molecular_simulations` (not installed) and needs
                                   Amber + precomputed MD trajectories
  structure_confidence          -> UNAVAILABLE offline: ChaiAgent needs a GPU + Chai-1 weights;
                                   TrajectoryAnalysisAgent._calculate_confidence is a shadowed stub
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Any, Optional

SBR_ROOT_ENV = "BINDER_ADAPTER_SBR_ROOT"

OBJECTIVE_VERSION = "sbr_pydantic_refactor_v1"

# The 8 filters SequenceQualityControl runs in __call__, in order.
_QC_CHECK_NAMES = [
    "multiplicity",
    "diversity",
    "repeat",
    "charge_ratio",
    "check_bad_motifs",
    "net_charge",
    "bad_terminus",
    "hydrophobicity",
]


def bootstrap_sbr_path(sbr_root: Optional[str] = None) -> Optional[str]:
    """Put the SBR checkout on sys.path. Idempotent.

    Precedence: explicit ``sbr_root`` arg, then the ``BINDER_ADAPTER_SBR_ROOT``
    environment variable (how the subprocess worker receives it).
    """
    root = sbr_root or os.environ.get(SBR_ROOT_ENV)
    if root and root not in sys.path:
        sys.path.insert(0, root)
    return root


def probe_capabilities() -> dict[str, Any]:
    """Import-probe the SBR modules this scorer can drive.

    Returns a dict with the imported classes (or None) and the import error
    string for any that failed. Call ``bootstrap_sbr_path`` first.
    """
    caps: dict[str, Any] = {
        "quality_control": None,
        "quality_control_error": "",
        "simple_energy": None,
        "simple_energy_error": "",
    }
    try:
        from struct_bio_reasoner.agents.computational_design.quality_control import (  # type: ignore
            SequenceQualityControl,
        )

        caps["quality_control"] = SequenceQualityControl
    except Exception as exc:  # pragma: no cover - environment dependent
        caps["quality_control_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from struct_bio_reasoner.agents.computational_design.energy import (  # type: ignore
            SimpleEnergy,
        )

        caps["simple_energy"] = SimpleEnergy
    except Exception as exc:  # pragma: no cover - environment dependent
        caps["simple_energy_error"] = f"{type(exc).__name__}: {exc}"

    return caps


def capabilities_summary(caps: dict[str, Any]) -> dict[str, Any]:
    """The JSON-serialisable subset of a capabilities probe (no class objects)."""
    return {
        "quality_control": caps["quality_control"] is not None,
        "quality_control_error": caps["quality_control_error"],
        "simple_energy": caps["simple_energy"] is not None,
        "simple_energy_error": caps["simple_energy_error"],
    }


# --- Objective computations --------------------------------------------------


def score_developability(sequence: str, caps: dict[str, Any]) -> dict[str, Any]:
    """Fraction of SequenceQualityControl filters the sequence passes.

    We drive the real QC object, then re-run each individual filter to get a
    graded score instead of the single pass/fail bool that __call__ returns.
    """
    qc_cls = caps["quality_control"]
    if qc_cls is None:
        return {
            "available": False,
            "reason": f"SequenceQualityControl import failed: {caps['quality_control_error']}",
        }
    if not sequence:
        return {"available": False, "reason": "empty candidate sequence"}

    qc = qc_cls()
    # __call__ populates qc.seq / qc.length / qc.counts / qc.pairs before running
    # checks, and prints the name of the first failing filter; silence that.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        overall_pass = bool(qc(sequence))

    per_check: dict[str, bool] = {}
    for name in _QC_CHECK_NAMES:
        fn = getattr(qc, name, None)
        if fn is None:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                per_check[name] = bool(fn())
        except Exception as exc:
            per_check[name] = False
            per_check[f"{name}__error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]

    graded = [v for k, v in per_check.items() if not k.endswith("__error")]
    score = (sum(1 for v in graded if v) / len(graded)) if graded else 0.0

    return {
        "available": True,
        "score": score,
        "evidence": {
            "source": "sbr.computational_design.quality_control.SequenceQualityControl",
            "overall_pass": overall_pass,
            "checks_passed": f"{sum(1 for v in graded if v)}/{len(graded)}",
            "per_check": {k: v for k, v in per_check.items()},
        },
    }


def _pairwise_identity(a: str, b: str) -> float:
    """Ungapped identity over the overlapping prefix, length-normalised."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / float(max(len(a), len(b)))


def score_specificity_proxy(sequence: str, reference_sequences: list[str]) -> dict[str, Any]:
    """Off-target risk proxy = novelty against a reference/off-target panel.

    score = 1 - max_identity(candidate, reference). Higher is better (more
    specific / less likely to cross-react with the reference panel).

    This is a sequence-side proxy. The embedding-based route
    (agents/embedding/sampling_agent.py) is unavailable: genslm_esm is not
    installed in this environment.
    """
    if not sequence:
        return {"available": False, "reason": "empty candidate sequence"}
    if not reference_sequences:
        return {
            "available": False,
            "reason": (
                "no specificity_reference_sequences supplied; sequence-novelty proxy "
                "requires an off-target panel (embedding agent unavailable: genslm_esm missing)"
            ),
        }

    identities = [(_pairwise_identity(sequence, ref), ref) for ref in reference_sequences if ref]
    if not identities:
        return {"available": False, "reason": "reference panel contained no usable sequences"}

    max_identity, closest = max(identities, key=lambda t: t[0])
    return {
        "available": True,
        "score": 1.0 - max_identity,
        "evidence": {
            "source": "adapter.sequence_novelty_proxy",
            "panel_size": len(identities),
            "max_identity": round(max_identity, 4),
            "closest_reference_prefix": closest[:24],
            "note": "embedding proxy unavailable (genslm_esm not installed)",
        },
    }


def score_binding_affinity(complex_pdb_path: str, caps: dict[str, Any]) -> dict[str, Any]:
    """Interface contact energy from SimpleEnergy. Requires a complex PDB.

    SimpleEnergy returns a negative-is-favourable contact count. We negate it so
    that, like every other objective here, higher is better.
    """
    energy_cls = caps["simple_energy"]
    if energy_cls is None:
        return {
            "available": False,
            "reason": f"SimpleEnergy import failed: {caps['simple_energy_error']}",
        }
    if not complex_pdb_path:
        return {
            "available": False,
            "reason": (
                "no complex_pdb_path supplied; SimpleEnergy scores a folded "
                "target-binder complex (chains A/B), not a bare sequence"
            ),
        }

    pdb = Path(complex_pdb_path)
    if not pdb.exists():
        return {"available": False, "reason": f"complex_pdb_path does not exist: {complex_pdb_path}"}

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            raw = float(energy_cls()(pdb))
    except Exception as exc:
        return {
            "available": False,
            "reason": f"SimpleEnergy raised {type(exc).__name__}: {exc}",
        }

    if raw != raw:  # NaN
        return {"available": False, "reason": "SimpleEnergy returned NaN"}

    return {
        "available": True,
        "score": -raw,  # SimpleEnergy: negative == favourable -> flip to maximize
        "evidence": {
            "source": "sbr.computational_design.energy.SimpleEnergy",
            "raw_energy": raw,
            "pdb": str(pdb),
            "note": "interface contact count within 5.0A between chain A and chain B; sign flipped to maximize",
        },
    }


def run_md_objectives(
    complex_pdb_path: str,
    md_config: Optional[dict[str, Any]] = None,
    framework_sequence: Optional[str] = None,
    candidate_sequence: Optional[str] = None,
) -> dict[str, Any]:
    """Run one OpenMM MD trajectory and derive BOTH physics-based objectives.

    Returns {"binding": {...}, "thermostability": {...}} where each is an
    availability dict shaped like the other scorers. A single trajectory feeds
    both, so this is called once per candidate (not once per objective).

    When ``framework_sequence`` + ``candidate_sequence`` are provided and
    ``md_config`` names a ``binder_chain``, the candidate's substitutions are
    threaded onto that chain so each candidate is simulated as its own mutant.
    Without a binder_chain the wild-type template is simulated as-is.

    MD is opt-in: callers set ``md_config`` (even ``{}`` for demo defaults) in the
    shard payload. When absent, MD objectives stay unavailable and no simulation
    runs. Requires OpenMM + pdbfixer and a multi-chain complex PDB.
    """
    from . import md_engine  # local import: OpenMM is heavy and optional

    unavailable = lambda reason: {  # noqa: E731
        "binding": {"available": False, "reason": reason},
        "thermostability": {"available": False, "reason": reason},
    }

    ok, import_err = md_engine.openmm_available()
    if not ok:
        return unavailable(f"OpenMM/pdbfixer not available: {import_err}")
    if not complex_pdb_path:
        return unavailable(
            "MD requires a complex_pdb_path (a folded target-binder complex); none supplied"
        )
    if not Path(complex_pdb_path).exists():
        return unavailable(f"complex_pdb_path does not exist: {complex_pdb_path}")

    md_config = md_config or {}
    # receptor_chains / ligand_chains / binder_chain are run_complex_md args, NOT
    # MDConfig fields; separate them so MDConfig(**...) doesn't choke.
    receptor_chains = md_config.get("receptor_chains")
    ligand_chains = md_config.get("ligand_chains")
    binder_chain = md_config.get("binder_chain")
    cfg_kwargs = {
        k: v
        for k, v in md_config.items()
        if k not in ("receptor_chains", "ligand_chains", "binder_chain")
    }
    try:
        cfg = md_engine.MDConfig(**cfg_kwargs)
    except TypeError as exc:
        return unavailable(f"invalid md_config: {exc}")

    try:
        result = md_engine.run_complex_md(
            complex_pdb_path,
            cfg,
            receptor_chains=receptor_chains,
            ligand_chains=ligand_chains,
            binder_chain=binder_chain,
            framework_sequence=framework_sequence,
            candidate_sequence=candidate_sequence,
        )
    except md_engine.MDUnavailable as exc:
        return unavailable(str(exc))
    except ValueError as exc:
        return unavailable(f"MD could not score this complex: {exc}")
    except Exception as exc:  # a failed simulation must not fabricate a score
        return unavailable(f"MD run raised {type(exc).__name__}: {exc}")

    binding = result["binding"]
    stability = result["stability"]
    prov = result["provenance"]

    # Binding: dG_bind is negative-favourable -> flip so higher is better.
    dg = binding["dg_bind_kcal_per_mol"]
    binding_obj = {
        "available": True,
        "score": -dg,
        "uncertainty": binding.get("dg_bind_std"),
        "evidence": {
            "source": "adapter.md_engine.openmm_mmgbsa",
            "method": "single-trajectory MM-GBSA (implicit solvent gbn2, ff14SB)",
            "dg_bind_kcal_per_mol": dg,
            "dg_bind_std": binding.get("dg_bind_std"),
            "n_frames": binding.get("n_frames"),
            "note": "sign flipped so higher score = more favourable binding",
            **{k: prov[k] for k in ("openmm_version", "platform", "receptor_chains",
                                    "ligand_chains", "prod_steps", "demo_scale",
                                    "binder_chain", "applied_mutations",
                                    "n_mutations_threaded")},
        },
    }

    # Thermostability proxy: lower fluctuation = more stable. Combine RMSF + Rg
    # drift + energy std into a single "instability" and negate to maximize.
    instability = (
        stability["mean_rmsf_nm"]
        + stability["rg_std_nm"]
        + 0.001 * stability["potential_energy_std"]
    )
    thermo_obj = {
        "available": True,
        "score": -instability,
        "evidence": {
            "source": "adapter.md_engine.trajectory_stability_proxy",
            "method": "conformational-stability proxy from implicit-solvent MD",
            "mean_rmsf_nm": stability["mean_rmsf_nm"],
            "rg_std_nm": stability["rg_std_nm"],
            "potential_energy_std": stability["potential_energy_std"],
            "n_frames": stability.get("n_frames"),
            "note": stability.get("note", ""),
            **{k: prov[k] for k in ("openmm_version", "platform", "prod_steps", "demo_scale")},
        },
    }
    return {"binding": binding_obj, "thermostability": thermo_obj}


def score_thermostability() -> dict[str, Any]:
    """Fallback when MD is not requested/available.

    With ``md_config`` in the shard payload and OpenMM present, thermostability
    is measured by ``run_md_objectives`` instead (a conformational-stability
    proxy from an OpenMM trajectory). This reason documents the no-MD path.
    """
    return {
        "available": False,
        "reason": (
            "thermostability not computed: no md_config supplied (OpenMM MD is opt-in). "
            "Supply md_config + a multi-chain complex_pdb_path to enable the "
            "conformational-stability proxy. StructBioReasoner's own MM-PBSA path stays "
            "unavailable: agents/molecular_dynamics/mmpbsa_agent.py needs the "
            "'molecular_simulations' package + Amber + precomputed trajectories"
        ),
    }


def score_structure_confidence() -> dict[str, Any]:
    return {
        "available": False,
        "reason": (
            "structure confidence requires folding: ChaiAgent needs a GPU and Chai-1 weights; "
            "TrajectoryAnalysisAgent._calculate_confidence is dead code (a second definition "
            "returning a constant 0.75 shadows the real RMSD heuristic), so it cannot be used"
        ),
    }


# --- Report assembly ---------------------------------------------------------


def _objective_value(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("available"):
        return {
            "score": float(result["score"]),
            "uncertainty": result.get("uncertainty"),
            "maximize": True,
            "available": True,
            "unavailable_reason": "",
        }
    return {
        "score": None,
        "uncertainty": None,
        "maximize": True,
        "available": False,
        "unavailable_reason": str(result.get("reason", "unspecified")),
    }


def score_one(
    candidate: dict[str, Any],
    target_id: str,
    run_id: str,
    reference_panel: list[str],
    objective_version: str,
    caps: dict[str, Any],
    md_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score a single candidate against a single target. Pure w.r.t. ``caps``.

    ``candidate`` keys: candidate_id, candidate_sequence, complex_pdb_path.
    Returns an ObjectiveReport dict (schema_version 0.2).

    When ``md_config`` is supplied AND a complex PDB is available, a single
    OpenMM MD trajectory provides a physics-based MM-GBSA binding free energy
    (overriding the crude SimpleEnergy contact count) and a conformational-
    stability proxy for thermostability. Otherwise those fall back to the
    static scorers / unavailable.
    """
    seq = str(candidate.get("candidate_sequence", ""))
    cid = str(candidate.get("candidate_id"))
    complex_pdb = str(candidate.get("complex_pdb_path", "") or "")
    framework = str(candidate.get("framework_sequence", "") or "")

    dev = score_developability(seq, caps)
    spec = score_specificity_proxy(seq, reference_panel)
    bind = score_binding_affinity(complex_pdb, caps)
    thermo = score_thermostability()
    conf = score_structure_confidence()

    # Opt-in physics: one MD run yields both binding + thermostability. It
    # overrides the static binding scorer (MM-GBSA dG >> contact count) only when
    # it actually produced a value; a failed/again-unavailable MD leaves the
    # SimpleEnergy result in place so we never regress an available objective.
    md_used = False
    if md_config is not None and complex_pdb:
        md = run_md_objectives(
            complex_pdb, md_config, framework_sequence=framework, candidate_sequence=seq
        )
        if md["binding"].get("available"):
            bind = md["binding"]
            md_used = True
        if md["thermostability"].get("available"):
            thermo = md["thermostability"]
            md_used = True
        elif thermo.get("available") is False and md["thermostability"].get("reason"):
            # surface the MD-specific reason rather than the generic no-md one
            thermo = md["thermostability"]

    results = {
        "developability": dev,
        "specificity_off_target_proxy": spec,
        "binding_affinity": bind,
        "thermostability": thermo,
        "structure_confidence": conf,
    }

    evidence = []
    tripped = []
    for name, res in results.items():
        if res.get("available") and res.get("evidence"):
            evidence.append({"objective": name, **res["evidence"]})
        if not res.get("available"):
            tripped.append(f"OBJECTIVE_UNAVAILABLE:{name}")
    if md_used:
        tripped.append("MD_OBJECTIVES_COMPUTED")

    return {
        "schema_version": "0.2",
        "report_id": f"{run_id}:{target_id}:{cid}",
        "candidate_id": cid,
        "target_id": target_id,
        "objective_version": objective_version,
        "binding_affinity": _objective_value(bind),
        "specificity_off_target_proxy": _objective_value(spec),
        "developability": _objective_value(dev),
        "thermostability": _objective_value(thermo),
        "structure_confidence": _objective_value(conf),
        "constraints_tripped": tripped,
        "evidence": evidence,
    }


def score_shard(payload: dict[str, Any], caps: dict[str, Any]) -> dict[str, Any]:
    """Score every candidate in a shard payload. Shared by both backends.

    ``payload`` schema: {run_id, target_id, objective_version,
                         specificity_reference_sequences, candidates:[...],
                         md_config?}.
    ``md_config`` (optional) enables the OpenMM MD objectives; ``{}`` uses demo
    defaults. Returns the worker output dict {ok, objective_version,
    capabilities, reports, worker_errors}.
    """
    import traceback

    worker_errors: list[str] = []
    reports: list[dict[str, Any]] = []

    try:
        run_id = str(payload.get("run_id", "run"))
        target_id = str(payload.get("target_id", "target"))
        objective_version = str(payload.get("objective_version", OBJECTIVE_VERSION))
        panel = [str(s) for s in payload.get("specificity_reference_sequences", []) if s]
        candidates = payload.get("candidates", [])
        md_config = payload.get("md_config")  # None => MD disabled

        for cand in candidates:
            try:
                reports.append(
                    score_one(
                        cand, target_id, run_id, panel, objective_version, caps, md_config
                    )
                )
            except Exception:
                worker_errors.append(
                    f"candidate {cand.get('candidate_id')} failed: {traceback.format_exc(limit=3)}"
                )
        ok = True
    except Exception:
        ok = False
        worker_errors.append(traceback.format_exc(limit=5))

    return {
        "ok": ok,
        "objective_version": OBJECTIVE_VERSION,
        "capabilities": capabilities_summary(caps),
        "reports": reports,
        "worker_errors": worker_errors,
    }
