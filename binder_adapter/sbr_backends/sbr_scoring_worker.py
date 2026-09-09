#!/usr/bin/env python3
"""StructBioReasoner scoring worker (Tier-1 subprocess boundary).

This script is executed as a subprocess by binder_adapter. It is a thin shell
around ``binder_adapter.sbr_backends.scoring`` — the shared scoring core that the
in-process backend (Tier 2) also calls. Keeping all scoring rules in that one
module is deliberate: a subprocess report and an in-process report for the same
candidate are then guaranteed identical, with no second copy to drift.

The subprocess still earns its keep: it isolates the SBR import graph
(MDAnalysis, academy, parsl, ...) from the host process, and makes each shard an
independent unit of work for a Slurm/PBS job array.

I/O contract:
  argv: --input <shard_input.json> --output <objective_reports.json>
  input  schema: {"run_id", "target_id", "objective_version",
                  "specificity_reference_sequences": [...], "candidates": [...]}
  output schema: {"ok": bool, "objective_version": str, "capabilities": {...},
                  "reports": [ObjectiveReport...], "worker_errors": [...]}
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

# The worker runs as a standalone script (spawned by subprocess), so it may not
# have the package on sys.path. Try the package-relative import first, then fall
# back to loading the sibling scoring.py directly.
try:
    from binder_adapter.sbr_backends import scoring  # type: ignore
except Exception:  # pragma: no cover - exercised only when run as a bare script
    import importlib.util
    import sys

    _here = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location(
        "binder_adapter_scoring", str(_here / "scoring.py")
    )
    scoring = importlib.util.module_from_spec(_spec)  # type: ignore[assignment]
    assert _spec and _spec.loader
    sys.modules["binder_adapter_scoring"] = scoring
    _spec.loader.exec_module(scoring)  # type: ignore[union-attr]


def main() -> int:
    ap = argparse.ArgumentParser(description="StructBioReasoner scoring worker")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # SBR checkout location arrives via BINDER_ADAPTER_SBR_ROOT (set by the backend).
    scoring.bootstrap_sbr_path()
    caps = scoring.probe_capabilities()

    try:
        payload = json.loads(Path(args.input).read_text())
        out = scoring.score_shard(payload, caps)
    except Exception:
        out = {
            "ok": False,
            "objective_version": scoring.OBJECTIVE_VERSION,
            "capabilities": scoring.capabilities_summary(caps),
            "reports": [],
            "worker_errors": [traceback.format_exc(limit=5)],
        }

    Path(args.output).write_text(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
