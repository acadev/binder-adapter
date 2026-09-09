from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapter_core import AdapterConfig, BinderAdapterCore

DEFAULT_SBR_ROOT = "/Users/ramanathana/Work/StructBioReasoner"


def _load(arg: str) -> Any:
    """Accept either an inline JSON string or a path to a JSON file."""
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(arg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="binder-adapter",
        description="Score Jnana binder hypotheses with StructBioReasoner and Pareto-select.",
    )
    ap.add_argument("--campaign-context", required=True, help="JSON string or path")
    ap.add_argument("--hypotheses", required=True, help="JSON string or path")
    ap.add_argument("--target-id", action="append", default=[], help="repeatable; overrides context")
    ap.add_argument("--run-id", default="run0")
    ap.add_argument("--max-survivors", type=int, default=10)
    ap.add_argument("--shard-size", type=int, default=16)
    ap.add_argument("--diversity-min-distance", type=float, default=0.0)
    ap.add_argument("--sbr-root", default=DEFAULT_SBR_ROOT)
    ap.add_argument("--python-executable", default=sys.executable)
    ap.add_argument(
        "--backend",
        choices=["subprocess", "in_process"],
        default="subprocess",
        help="subprocess (Tier 1, process isolation) or in_process (Tier 2, direct SBR calls)",
    )
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--output", default=None, help="write result JSON here as well as stdout")
    args = ap.parse_args(argv)

    campaign_context = _load(args.campaign_context)
    hypotheses = _load(args.hypotheses)

    core = BinderAdapterCore(
        AdapterConfig(
            sbr_root=args.sbr_root,
            python_executable=args.python_executable,
            shard_size=args.shard_size,
            max_survivors=args.max_survivors,
            diversity_min_distance=args.diversity_min_distance,
            artifact_dir=args.artifact_dir,
            backend=args.backend,
        )
    )

    result = core.score_and_select(
        hypotheses=hypotheses,
        campaign_context=campaign_context,
        run_id=args.run_id,
        target_ids=args.target_id or None,
    )

    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0 if not result.get("scoring_error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
