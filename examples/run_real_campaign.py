#!/usr/bin/env python3
"""End-to-end binder-design campaign on a REAL protein complex.

This is the realistic demo: it takes an actual crystal structure of a
protein-protein complex (1YCR = MDM2 receptor + p53 transactivation peptide),
treats the p53 peptide as the binder framework, generates several point-mutant
binder candidates, and runs the full binder-adapter pipeline end to end:

    extract -> mutation-budget gate -> OpenMM MD scoring -> Pareto selection

Every candidate is scored on real physics:
  * binding_affinity  = single-trajectory MM-GBSA dG_bind (OpenMM, ff14SB, gbn2)
  * thermostability   = conformational-stability proxy from the same trajectory
  * developability    = StructBioReasoner SequenceQualityControl (8 filters)
  * specificity proxy = sequence novelty vs. the wild-type framework panel

No Amber, no molecular_simulations, no GPU required (demo scale runs on CPU).

Usage:
    python examples/run_real_campaign.py                 # fetch 1YCR, demo-scale MD
    python examples/run_real_campaign.py --pdb-id 1YCR --n-candidates 4
    python examples/run_real_campaign.py --complex-pdb my_complex.pdb \
        --binder-chain B --n-candidates 5 --prod-steps 4000

Requires: openmm, pdbfixer, and a StructBioReasoner checkout (for the sequence
QC objective). Set BINDER_ADAPTER_SBR_ROOT if it is not at the default path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Make the adapter + SBR importable when run straight from the repo.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SBR = os.environ.get("BINDER_ADAPTER_SBR_ROOT", "/Users/ramanathana/Work/StructBioReasoner")
if Path(_SBR).is_dir() and _SBR not in sys.path:
    sys.path.insert(0, _SBR)

AA = "ACDEFGHIKLMNPQRSTVWY"
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def prepare_complex(pdb_id: str | None, complex_pdb: str | None, out_pdb: Path):
    """Load + repair a complex, return (path, chain_ids). Fetches by id if needed."""
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    if complex_pdb:
        fixer = PDBFixer(filename=complex_pdb)
    else:
        print(f"[prep] fetching {pdb_id} from the PDB ...")
        fixer = PDBFixer(pdbid=pdb_id)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    with open(out_pdb, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    chains = [c.id for c in fixer.topology.chains()]
    return out_pdb, chains, fixer


def framework_sequence_of_chain(fixer, chain_id: str) -> str:
    """One-letter sequence of a chain from the fixed structure."""
    seq = []
    for chain in fixer.topology.chains():
        if chain.id != chain_id:
            continue
        for res in chain.residues():
            aa = THREE_TO_ONE.get(res.name.upper())
            if aa:
                seq.append(aa)
    return "".join(seq)


def make_candidates(framework: str, n: int, min_frac: float, max_frac: float, seed: int):
    """Generate n binder candidates inside the mutation budget."""
    rng = random.Random(seed)
    L = len(framework)
    lo, hi = max(1, int(min_frac * L)), max(1, int(max_frac * L))
    candidates = []
    for i in range(n):
        n_mut = rng.randint(lo, hi)
        s = list(framework)
        for p in rng.sample(range(L), n_mut):
            s[p] = rng.choice([a for a in AA if a != s[p]])
        candidates.append(
            {"hypothesis_id": f"cand_{i+1}", "candidate_sequence": "".join(s)}
        )
    return candidates


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-id", default="1YCR", help="PDB id to fetch (default 1YCR: MDM2/p53)")
    ap.add_argument("--complex-pdb", default=None, help="local complex PDB instead of fetching")
    ap.add_argument("--binder-chain", default=None,
                    help="chain to treat as the binder framework (default: last chain)")
    ap.add_argument("--n-candidates", type=int, default=4)
    ap.add_argument("--prod-steps", type=int, default=2000, help="MD production steps (demo default)")
    ap.add_argument("--equil-steps", type=int, default=500)
    ap.add_argument("--platform", default="CPU", help="OpenMM platform: CPU | OpenCL | CUDA")
    ap.add_argument("--max-survivors", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--artifact-dir", default=str(_REPO / "runs" / "real_campaign"))
    ap.add_argument("--sbr-root", default=_SBR)
    args = ap.parse_args(argv)

    from binder_adapter.adapter_core import AdapterConfig, BinderAdapterCore

    workdir = Path(args.artifact_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    fixed_pdb = workdir / "complex_fixed.pdb"

    complex_path, chains, fixer = prepare_complex(args.pdb_id, args.complex_pdb, fixed_pdb)
    print(f"[prep] complex ready: {complex_path}  chains={chains}")
    if len(chains) < 2:
        print("ERROR: need a >=2-chain complex to score binding.", file=sys.stderr)
        return 2

    binder_chain = args.binder_chain or chains[-1]
    receptor_chains = [c for c in chains if c != binder_chain]
    framework = framework_sequence_of_chain(fixer, binder_chain)
    print(f"[prep] binder chain={binder_chain} (framework len={len(framework)}): {framework}")
    print(f"[prep] receptor chains={receptor_chains}")

    candidates = make_candidates(framework, args.n_candidates, 0.20, 0.50, args.seed)
    print(f"[gen] {len(candidates)} candidates generated inside the 20-50% mutation budget")

    campaign_context = {
        "framework_sequence": framework,
        "target_ids": [args.pdb_id],
        "complex_pdb_path": str(complex_path),
        "specificity_reference_sequences": [framework],  # WT is the off-target to avoid
        "md_config": {
            "equil_steps": args.equil_steps,
            "prod_steps": args.prod_steps,
            "sample_interval": max(100, args.prod_steps // 10),
            "platform": args.platform,
            "binder_chain": binder_chain,
            "receptor_chains": receptor_chains,
            "ligand_chains": [binder_chain],
        },
    }

    core = BinderAdapterCore(
        AdapterConfig(
            sbr_root=args.sbr_root,
            backend="in_process",       # run MD in this interpreter
            artifact_dir=str(workdir),
            max_survivors=args.max_survivors,
        )
    )

    print(f"[md] running OpenMM MD campaign (prod_steps={args.prod_steps}, platform={args.platform}) ...")
    result = core.score_and_select(
        hypotheses=candidates,
        campaign_context=campaign_context,
        run_id="real_campaign",
    )

    out_json = workdir / "campaign_result.json"
    out_json.write_text(json.dumps(result, indent=2))

    print("\n===================== CAMPAIGN RESULT =====================")
    print(f"active objectives : {result['active_objectives']}")
    print(f"unavailable       : {list(result['unavailable_objectives'])}")
    print(f"mutation budget   : {result['mutation_budget']}")
    print(f"invalid (budget)  : {[i['candidate_id'] for i in result['invalid_candidates']]}")
    print(f"\nRanked survivors ({len(result['rankings'])}):")
    for r in result["rankings"]:
        rep = r["objective_reports"][0]
        def _fmt(objname, prec):
            v = rep[objname].get("score")
            return f"{v:.{prec}f}" if isinstance(v, (int, float)) else "n/a"
        print(f"  #{r['rank']} {r['candidate_id']:8s} "
              f"pareto_front={r['pareto_front_id']}  "
              f"binding(-dG)={_fmt('binding_affinity', 2)}  "
              f"thermo={_fmt('thermostability', 4)}  "
              f"dev={_fmt('developability', 3)}")
    print(f"\nfull result -> {out_json}")
    print("===========================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
