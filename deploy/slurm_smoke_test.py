#!/usr/bin/env python3
"""SLURM smoke test for binder-adapter's Academy+Parsl scale-out path.

This is the minimal "does scale-out actually work on this machine" check you run
FIRST on a new cluster, before committing an allocation to a real campaign. It:

  1. builds the site's Parsl config (via --site) and loads it,
  2. runs a trivial Parsl @python_app on the ALLOCATED compute (proves the
     provider/launcher can place work on nodes, not just the login node),
  3. runs a real one-shard scoring through the AcademyScoringBackend on a couple
     of synthetic candidates (proves the SBR stack imports and scores on the
     workers), and
  4. prints a clear PASS/FAIL with timings.

It deliberately uses TINY work so it finishes in a couple of minutes inside a
debug-queue allocation. It does NOT need internet (no generation step).

Exit code 0 = smoke passed; non-zero = failed (so an sbatch script can gate on it).

Usage (inside an allocation / from the sbatch script below):
    python deploy/slurm_smoke_test.py --site leonardo
    python deploy/slurm_smoke_test.py --site local          # laptop dry-run
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SBR = os.environ.get("BINDER_ADAPTER_SBR_ROOT", "/opt/StructBioReasoner")
if Path(_SBR).is_dir() and _SBR not in sys.path:
    sys.path.insert(0, _SBR)

FW = "SQETFSDLWKLLPEN"
AA = "ACDEFGHIKLMNPQRSTVWY"


def _mutate(seq: str, n: int, seed: int) -> str:
    r = random.Random(seed)
    s = list(seq)
    for p in r.sample(range(len(seq)), n):
        s[p] = r.choice([a for a in AA if a != s[p]])
    return "".join(s)


def _p(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="local", help="parsl site (local/leonardo/polaris)")
    ap.add_argument("--sbr-root", default=_SBR)
    ap.add_argument("--artifact-dir", default=os.environ.get("SCRATCH", "/tmp") + "/binder_adapter_smoke")
    ap.add_argument("--n", type=int, default=2, help="candidates to score")
    args = ap.parse_args(argv)

    Path(args.artifact_dir).mkdir(parents=True, exist_ok=True)
    _p(f"site={args.site} sbr_root={args.sbr_root} artifact_dir={args.artifact_dir}")

    failures = []

    # -- step 1: build + load the site Parsl config ---------------------------
    try:
        from examples.parsl_configs import SITE_CONFIGS
        import parsl

        if args.site not in SITE_CONFIGS:
            _p(f"FAIL: unknown site {args.site!r}; choices {list(SITE_CONFIGS)}")
            return 2
        cfg = SITE_CONFIGS[args.site]()
        prov = cfg.executors[0].provider
        _p(f"config OK: executor={type(cfg.executors[0]).__name__} provider={type(prov).__name__}")
        parsl.load(cfg)
    except Exception as exc:
        import traceback
        _p("FAIL: could not load Parsl config\n" + traceback.format_exc())
        return 2

    # -- step 2: trivial task ON the allocation -------------------------------
    try:
        from parsl import python_app

        @python_app
        def _where(x):
            import socket
            import os as _os

            return f"{x} ran on {socket.gethostname()} (pid {_os.getpid()})"

        t0 = time.time()
        msg = _where(42).result()
        _p(f"parsl task OK in {time.time()-t0:.1f}s: {msg}")
    except Exception as exc:
        import traceback
        _p("FAIL: trivial Parsl task did not run on the allocation\n" + traceback.format_exc())
        failures.append("parsl_task")

    # -- step 3: real one-shard scoring via the Academy backend ---------------
    try:
        from binder_adapter.sbr_backends.academy_backend import (
            AcademyBackendConfig,
            AcademyScoringBackend,
        )
        from binder_adapter.schemas import CandidateArtifact
        from binder_adapter.extraction import extract_candidate_artifacts_from_jnana

        ctx = {"framework_sequence": FW, "target_ids": ["SMOKE"]}
        hyps = [
            {"hypothesis_id": f"s{i}", "candidate_sequence": _mutate(FW, 4, 200 + i)}
            for i in range(args.n)
        ]
        candidates = extract_candidate_artifacts_from_jnana(hyps, ctx)
        candidates = [c for c in candidates if c.candidate_sequence]

        # Reuse the already-loaded Parsl DFK (pass a factory that returns current).
        backend = AcademyScoringBackend(
            AcademyBackendConfig(
                sbr_root=args.sbr_root,
                parsl_config_factory=lambda: __import__("parsl").dfk().config,
                artifact_dir=args.artifact_dir,
            )
        )
        # DFK already loaded in step 1/2; mark backend as using it.
        backend._dfk = __import__("parsl").dfk()

        t0 = time.time()
        reports, meta = backend.score_candidates_batch(
            campaign_context=ctx,
            target_id="SMOKE",
            candidates_batch=candidates,
            run_id="smoke",
            shard_id=0,
        )
        _p(f"scoring shard OK in {time.time()-t0:.1f}s: "
           f"backend={meta.get('backend')} ok={meta.get('ok')} reports={len(reports)}")
        if not reports:
            _p("WARN: shard produced no reports (check SBR checkout on workers)")
            failures.append("no_reports")
        else:
            r0 = reports[0].to_dict() if hasattr(reports[0], "to_dict") else reports[0]
            active = [k for k in ("developability", "specificity_off_target_proxy")
                      if isinstance(r0, dict) and r0.get(k, {}).get("available")]
            _p(f"first report active objectives: {active}")
    except Exception as exc:
        import traceback
        _p("FAIL: Academy scoring shard errored\n" + traceback.format_exc())
        failures.append("scoring")

    # -- teardown -------------------------------------------------------------
    try:
        import parsl

        parsl.dfk().cleanup()
        parsl.clear()
    except Exception:
        pass

    if failures:
        _p(f"RESULT: FAIL ({', '.join(failures)})")
        return 1
    _p("RESULT: PASS — scale-out path works on this machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
