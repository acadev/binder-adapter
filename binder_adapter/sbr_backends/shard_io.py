"""Shared shard I/O helpers used by both scoring backends.

Building the shard-input payload (panel assembly, complex-PDB defaulting) and
laying out the per-shard artifact directory must be identical across the
subprocess and in-process backends, so it lives here rather than in either one.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, List, Optional

from ..schemas import CandidateArtifact


def shard_dir(
    artifact_dir: Optional[str], run_id: str, target_id: str, shard_id: int
) -> Path:
    """Resolve the directory for one shard's artifacts.

    With an artifact_dir, artifacts are auditable at a stable path. Without one,
    a throwaway temp dir is used (in-process callers that don't persist still get
    a valid path for metadata).
    """
    if artifact_dir:
        d = Path(artifact_dir) / run_id / f"target_{target_id}" / f"shard_{shard_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(tempfile.mkdtemp(prefix=f"sbr_shard_{target_id}_{shard_id}_"))


def build_shard_input(
    run_id: str,
    target_id: str,
    candidates: List[CandidateArtifact],
    campaign_context: dict[str, Any],
    extra_reference_sequences: Optional[List[str]] = None,
) -> dict[str, Any]:
    """Assemble the JSON payload consumed by ``scoring.score_shard``.

    The specificity panel is the union of: backend-config references, campaign
    references, and the framework itself (a binder that barely moved from the
    framework is not a specific new binder). Complex-PDB path falls back from the
    candidate to the campaign default.
    """
    panel = list(extra_reference_sequences or [])
    panel.extend(campaign_context.get("specificity_reference_sequences", []) or [])
    fw = campaign_context.get("framework_sequence")
    if fw:
        panel.append(str(fw))

    payload: dict[str, Any] = {
        "run_id": run_id,
        "target_id": target_id,
        "objective_version": campaign_context.get(
            "objective_version", "sbr_pydantic_refactor_v1"
        ),
        "specificity_reference_sequences": list(dict.fromkeys(panel)),
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "candidate_sequence": c.candidate_sequence,
                "framework_sequence": c.framework_sequence,
                "complex_pdb_path": c.complex_pdb_path
                or campaign_context.get("complex_pdb_path", "")
                or "",
            }
            for c in candidates
        ],
    }
    # Opt-in OpenMM MD objectives (MM-GBSA binding + thermostability proxy).
    # Present only when the campaign requests it, so the default fast path is
    # unchanged and MD never runs implicitly.
    md_config = campaign_context.get("md_config")
    if md_config is not None:
        payload["md_config"] = md_config
    return payload
