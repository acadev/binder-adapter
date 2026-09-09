#!/usr/bin/env python3
"""Portable, scale-out scoring entrypoint: candidates.jsonl -> rankings.json.

This is the deployable driver for running binder-adapter scoring at scale on any
supported site. It is deliberately decoupled from generation: an upstream step
(Jnana+ARGO on a login/service node, or any generator) produces a candidates
file; this entrypoint fans the scoring out across the site's compute via Academy
+ Parsl and writes a ranked result.

Portability: the ONLY thing that changes between a laptop, CINECA Leonardo, and
ALCF Polaris is ``--site`` (which selects a Parsl config factory from
``examples/parsl_configs.py``). The agent code and scoring core are identical
everywhere.

It drives the scoring through the Academy ``ScoringAgent`` via a ``Manager`` +
``LocalExchangeFactory``, exercising the real agent path (not just the backend),
so the same invocation works when the agent is later launched onto a remote
Globus Compute endpoint.

Input format (JSON Lines), one candidate per line:
    {"hypothesis_id": "c0", "candidate_sequence": "SQ...EN", "complex_pdb_path": "..."}
``candidate_sequence`` is required; ``complex_pdb_path`` is optional (enables
binding/MD objectives). You can also pass free-text Jnana hypotheses with a
``content`` field — extraction will pull sequences framework-aware.

Usage:
    python examples/run_academy_campaign.py \
        --candidates candidates.jsonl \
        --framework SQETFSDLWKLLPEN --target MDM2 \
        --site local --out rankings.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SBR = os.environ.get("BINDER_ADAPTER_SBR_ROOT", "/Users/ramanathana/Work/StructBioReasoner")
if Path(_SBR).is_dir() and _SBR not in sys.path:
    sys.path.insert(0, _SBR)


def load_candidates(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


async def _run(args) -> dict:
    from academy.exchange import LocalExchangeFactory
    from academy.manager import Manager
    from concurrent.futures import ThreadPoolExecutor

    from binder_adapter.academy_agent import ScoringAgent
    from examples.parsl_configs import SITE_CONFIGS

    if args.site not in SITE_CONFIGS:
        raise SystemExit(f"unknown --site {args.site!r}; choices: {list(SITE_CONFIGS)}")
    parsl_factory = SITE_CONFIGS[args.site]

    hypotheses = load_candidates(args.candidates)
    campaign_context: dict = {
        "framework_sequence": args.framework,
        "target_ids": [args.target],
    }
    if args.complex_pdb:
        campaign_context["complex_pdb_path"] = args.complex_pdb
    if args.md:
        campaign_context["md_config"] = {
            "equil_steps": args.equil_steps,
            "prod_steps": args.prod_steps,
            "sample_interval": max(100, args.prod_steps // 10),
            "platform": args.md_platform,
            "binder_chain": args.binder_chain,
            "receptor_chains": None,
            "ligand_chains": [args.binder_chain],
        }

    print(f"[academy] site={args.site} candidates={len(hypotheses)} "
          f"md={'on' if args.md else 'off'} platform={args.md_platform}")

    # Drive the agent through a Manager + local exchange. Scoring still fans out
    # via Parsl inside the agent, so this is real parallelism.
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(max_workers=2),
    ) as manager:
        handle = await manager.launch(
            ScoringAgent,
            kwargs=dict(
                sbr_root=_SBR,
                parsl_config_factory=parsl_factory,
                shard_size=args.shard_size,
                artifact_dir=args.artifact_dir,
            ),
        )
        caps = await handle.capabilities()
        print(f"[academy] worker capabilities: {caps}")
        result = await handle.score_and_select(
            hypotheses=hypotheses,
            campaign_context=campaign_context,
            run_id=args.run_id,
            target_ids=[args.target],
        )
        try:
            await handle.shutdown_parsl()
        except Exception:
            pass
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="candidates.jsonl path")
    ap.add_argument("--framework", required=True, help="framework (wild-type) sequence")
    ap.add_argument("--target", default="TARGET", help="target id")
    ap.add_argument("--site", default="local", help="parsl site config (local/leonardo/polaris)")
    ap.add_argument("--out", default="rankings.json")
    ap.add_argument("--run-id", default="academy_campaign")
    ap.add_argument("--shard-size", type=int, default=8)
    ap.add_argument("--artifact-dir", default=str(_REPO / "runs" / "academy"))
    ap.add_argument("--complex-pdb", default=None)
    ap.add_argument("--md", action="store_true", help="enable OpenMM MD objectives")
    ap.add_argument("--md-platform", default="CPU", help="CPU or CUDA")
    ap.add_argument("--binder-chain", default="B")
    ap.add_argument("--equil-steps", type=int, default=300)
    ap.add_argument("--prod-steps", type=int, default=1500)
    args = ap.parse_args(argv)

    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    result = asyncio.run(_run(args))

    Path(args.out).write_text(json.dumps(result, indent=2))

    print("\n===================== ACADEMY CAMPAIGN RESULT =====================")
    print(f"active objectives : {result['active_objectives']}")
    print(f"unavailable       : {list(result['unavailable_objectives'])}")
    print(f"invalid (budget)  : {[i['candidate_id'] for i in result['invalid_candidates']]}")
    print(f"shards scored     : {len(result['scoring_metadata'])} "
          f"(backend={result['scoring_metadata'][0]['backend'] if result['scoring_metadata'] else 'n/a'})")
    print(f"\nRanked survivors ({len(result['rankings'])}):")
    for r in result["rankings"]:
        rep = r["objective_reports"][0]

        def _fmt(o, p):
            v = rep[o].get("score")
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "n/a"

        print(f"  #{r['rank']} {r['candidate_id']:12s} front={r['pareto_front_id']} "
              f"binding={_fmt('binding_affinity', 2)} dev={_fmt('developability', 3)} "
              f"spec={_fmt('specificity_off_target_proxy', 3)} thermo={_fmt('thermostability', 4)}")
    print(f"\nfull result -> {args.out}")
    print("===================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
