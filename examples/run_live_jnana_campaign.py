#!/usr/bin/env python3
"""LIVE end-to-end: Jnana (ARGO LLM) generates -> binder-adapter scores + ranks.

This is the full-loop demo the earlier runs were building toward. It exercises the
real integration, not a canned candidate list:

  1. Jnana's ProtoGnosis CoScientist (real GenerationAgents) generates binder
     hypotheses by calling the ARGO LLM backend live, through the adapter's
     fence-tolerant LLM wrapper (binder_adapter.jnana_argo_llm).
  2. The adapter extracts candidate sequences from those hypotheses
     (extraction.extract_candidate_artifacts_from_jnana).
  3. The adapter scores + Pareto-ranks the extracted candidates against a target
     (BinderAdapterCore.score_and_select) using StructBioReasoner objectives, and
     optionally the OpenMM MD objectives (--md).

It reports honestly: hypotheses whose text carries no usable sequence are counted
as extraction failures rather than faked.

Prereqs: ARGO reachable at --argo-base (default localhost:63375/v1), a Jnana
checkout and a StructBioReasoner checkout. Nothing in Jnana or SBR is patched;
the ARGO wiring and fenced-JSON tolerance live entirely in the adapter.

Usage:
    python examples/run_live_jnana_campaign.py --n 4
    python examples/run_live_jnana_campaign.py --n 6 --md --complex-pdb complex.pdb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_JNANA = os.environ.get("BINDER_ADAPTER_JNANA_ROOT", "/Users/ramanathana/Work/Jnana")
_SBR = os.environ.get("BINDER_ADAPTER_SBR_ROOT", "/Users/ramanathana/Work/StructBioReasoner")
for p in (_JNANA, _SBR):
    if Path(p).is_dir() and p not in sys.path:
        sys.path.insert(0, p)

# p53 transactivation peptide (the natural MDM2 binder) as the design framework.
P53_FRAMEWORK = "SQETFSDLWKLLPEN"


def _hyp_to_dict(h):
    if isinstance(h, dict):
        return h
    d = getattr(h, "__dict__", {})
    return {
        "hypothesis_id": d.get("hypothesis_id"),
        "content": d.get("content", ""),
        "summary": d.get("summary", ""),
        "agent_id": d.get("agent_id", ""),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4, help="hypotheses to generate")
    ap.add_argument("--model", default="argo:gpt-4o")
    ap.add_argument("--argo-base", default="http://localhost:63375/v1")
    ap.add_argument("--framework", default=P53_FRAMEWORK)
    ap.add_argument("--target-id", default="MDM2")
    ap.add_argument("--md", action="store_true", help="enable OpenMM MD objectives")
    ap.add_argument("--complex-pdb", default=None, help="complex PDB for MD/binding")
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--gen-timeout", type=int, default=240)
    ap.add_argument("--artifact-dir", default=str(_REPO / "runs" / "live_jnana"))
    args = ap.parse_args(argv)

    workdir = Path(args.artifact_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    # --- 1. LIVE generation via Jnana + ARGO ---------------------------------
    from binder_adapter.jnana_argo_llm import build_argo_coscientist

    print(f"[gen] building Jnana CoScientist on ARGO ({args.model}) ...")
    cs = build_argo_coscientist(model=args.model, base_url=args.argo_base, max_workers=2)

    goal = (
        f"Design short peptide binders of about {len(args.framework)} amino acids that bind the "
        f"{args.target_id} p53-binding cleft with high affinity and specificity. The wild-type "
        f"framework peptide is {args.framework}. Propose variants that each carry BETWEEN 3 AND 7 "
        f"point substitutions relative to the framework (i.e. change 3-7 of the {len(args.framework)} "
        f"positions - not fewer, not more). For every proposed binder, give its explicit "
        f"single-letter amino-acid sequence on its own, clearly labelled 'candidate_sequence: <SEQ>'."
    )
    cs.set_research_goal(goal)
    cs.start()
    print(f"[gen] generating {args.n} hypotheses via ARGO (live LLM calls) ...")
    t0 = time.time()
    cs.generate_hypotheses(count=args.n)
    cs.wait_for_completion(timeout=args.gen_timeout)
    hyps = [_hyp_to_dict(h) for h in cs.get_all_hypotheses()]
    print(f"[gen] got {len(hyps)} hypotheses in {time.time()-t0:.1f}s")
    for h in hyps:
        print(f"       - {h['hypothesis_id']}: {str(h.get('summary',''))[:90]}")

    (workdir / "hypotheses.json").write_text(json.dumps(hyps, indent=2))

    # --- 2. Extraction (honest: prose w/o a sequence -> extraction failure) ---
    from binder_adapter.extraction import extract_candidate_artifacts_from_jnana

    campaign_context = {
        "framework_sequence": args.framework,
        "target_ids": [args.target_id],
    }
    if args.complex_pdb:
        campaign_context["complex_pdb_path"] = args.complex_pdb
    if args.md:
        campaign_context["md_config"] = {
            "equil_steps": 300, "prod_steps": 1500, "sample_interval": 150,
            "platform": "CPU", "binder_chain": args.binder_chain,
            "receptor_chains": None, "ligand_chains": [args.binder_chain],
        }

    artifacts = extract_candidate_artifacts_from_jnana(hyps, campaign_context)
    extracted = [a for a in artifacts if a.candidate_sequence]
    print(f"[extract] {len(extracted)}/{len(artifacts)} hypotheses yielded a usable sequence")
    for a in artifacts:
        tag = a.candidate_sequence[:30] if a.candidate_sequence else "<no sequence>"
        print(f"          {a.candidate_id}: conf={a.extraction_confidence:.2f} seq={tag}")

    if not extracted:
        print("[done] no sequences extracted from the live hypotheses; nothing to score.")
        print("       (This is honest output — the LLM proposed strategies, not explicit sequences.)")
        return 0

    # --- 3. Adapter scoring + Pareto selection --------------------------------
    from binder_adapter.adapter_core import AdapterConfig, BinderAdapterCore

    # Feed the extracted candidates back as hypotheses with structured sequences.
    scored_hyps = [
        {"hypothesis_id": a.candidate_id, "candidate_sequence": a.candidate_sequence}
        for a in extracted
    ]
    core = BinderAdapterCore(
        AdapterConfig(sbr_root=_SBR, backend="in_process", artifact_dir=str(workdir))
    )
    print(f"[score] scoring {len(scored_hyps)} candidates (md={'on' if args.md else 'off'}) ...")
    result = core.score_and_select(scored_hyps, campaign_context, run_id="live_jnana")

    (workdir / "campaign_result.json").write_text(json.dumps(result, indent=2))

    print("\n===================== LIVE CAMPAIGN RESULT =====================")
    print(f"active objectives : {result['active_objectives']}")
    print(f"unavailable       : {list(result['unavailable_objectives'])}")
    print(f"invalid (budget)  : {[i['candidate_id'] for i in result['invalid_candidates']]}")
    print(f"\nRanked survivors ({len(result['rankings'])}):")
    for r in result["rankings"]:
        rep = r["objective_reports"][0]
        def _fmt(o, p):
            v = rep[o].get("score")
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "n/a"
        print(f"  #{r['rank']} {r['candidate_id']:10s} front={r['pareto_front_id']} "
              f"binding={_fmt('binding_affinity',2)} dev={_fmt('developability',3)} "
              f"spec={_fmt('specificity_off_target_proxy',3)} thermo={_fmt('thermostability',4)}")
    print(f"\nfull result -> {workdir / 'campaign_result.json'}")
    print("================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
