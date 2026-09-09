"""Jnana integration: drop-in multi-objective ranking backed by binder-adapter.

Front-end 2 of the two-front-end design. `patch_ranking_agent` monkey-patches
`RankingAgent._rank_hypotheses` so ProtoGnosis tournaments rank binder
candidates on real StructBioReasoner objectives instead of an LLM's opinion,
without editing Jnana's source tree.

Usage inside a Jnana process:

    from binder_adapter.jnana_hook import patch_ranking_agent
    patch_ranking_agent(campaign_context={
        "framework_sequence": "...",
        "target_ids": ["TARGET_A"],
    }, sbr_root="/path/to/StructBioReasoner")

Rejected candidates (mutation budget) are dropped from the rankings and
surfaced via `get_last_resample_request()` so the generation layer can
resample the shortfall. That is the reject-and-resample policy: never repair.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .adapter_core import AdapterConfig, BinderAdapterCore

logger = logging.getLogger(__name__)

_LAST_RESAMPLE: dict[str, Any] = {}


def get_last_resample_request() -> dict[str, Any]:
    """Resample state from the most recent ranking call.

    Keys: needs_resample, shortfall, invalid_candidate_ids, invalid_details.
    """
    return dict(_LAST_RESAMPLE)


def _hypothesis_to_dict(hyp: Any) -> dict[str, Any]:
    """Normalise a Jnana ResearchHypothesis into the adapter's input shape."""
    out: dict[str, Any] = {
        "hypothesis_id": getattr(hyp, "hypothesis_id", None) or getattr(hyp, "id", None),
        "content": getattr(hyp, "content", "") or "",
    }
    # Opportunistically forward structured fields when Jnana starts emitting them.
    for attr in ("candidate_sequence", "sequence", "complex_pdb_path", "strategy", "agent_id"):
        val = getattr(hyp, attr, None)
        if val:
            out[attr] = val
    meta = getattr(hyp, "metadata", None)
    if isinstance(meta, dict):
        for key in ("candidate_sequence", "sequence", "complex_pdb_path"):
            if key in meta and key not in out:
                out[key] = meta[key]
    return out


def build_rankings(
    hypotheses: list[Any],
    core: BinderAdapterCore,
    campaign_context: dict[str, Any],
    run_id: str,
    max_survivors: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Produce Jnana-shaped ranking dicts from adapter decisions.

    The returned dicts keep the keys Jnana's RankingAgent already emits
    (hypothesis_id, rank, score, justification) and add the multi-objective
    detail alongside, so existing consumers keep working.
    """
    payload = [_hypothesis_to_dict(h) for h in hypotheses]
    result = core.score_and_select(
        hypotheses=payload,
        campaign_context=campaign_context,
        run_id=run_id,
        max_survivors=max_survivors,
    )

    _LAST_RESAMPLE.clear()
    _LAST_RESAMPLE.update(
        {
            "needs_resample": result.get("needs_resample", False),
            "shortfall": result.get("resample_shortfall", 0),
            "invalid_candidate_ids": [i["candidate_id"] for i in result.get("invalid_candidates", [])],
            "invalid_details": result.get("invalid_candidates", []),
            "active_objectives": result.get("active_objectives", []),
            "unavailable_objectives": result.get("unavailable_objectives", {}),
        }
    )

    if result.get("scoring_error"):
        logger.error("binder-adapter scoring failed: %s", result["scoring_error"])
        return []

    by_id = {getattr(h, "hypothesis_id", None) or getattr(h, "id", None): h for h in hypotheses}

    rankings: list[dict[str, Any]] = []
    for d in result.get("rankings", []):
        cid = d["candidate_id"]
        hyp = by_id.get(cid)
        content = getattr(hyp, "content", "") if hyp is not None else ""
        rankings.append(
            {
                "hypothesis_id": cid,
                "rank": d["rank"],
                "score": d["score_summary_utility"],
                "justification": d["justification"],
                "content_preview": (content[:100] + "...") if content else "",
                # multi-objective detail
                "pareto_front_id": d["pareto_front_id"],
                "active_objectives": d["active_objectives"],
                "objective_reports": d["objective_reports"],
            }
        )

    if result.get("invalid_candidates"):
        logger.info(
            "binder-adapter rejected %d candidate(s) on the mutation budget; "
            "resample shortfall=%s",
            len(result["invalid_candidates"]),
            result.get("resample_shortfall"),
        )

    return rankings


def patch_ranking_agent(
    campaign_context: dict[str, Any],
    sbr_root: str,
    ranking_agent_cls: Any = None,
    run_id_prefix: str = "jnana",
    **adapter_kwargs: Any,
) -> Callable[[], None]:
    """Replace RankingAgent._rank_hypotheses with adapter-backed ranking.

    Returns a callable that restores the original method.
    """
    if ranking_agent_cls is None:
        from jnana.protognosis.agents.ranking_agent import RankingAgent  # type: ignore

        ranking_agent_cls = RankingAgent

    core = BinderAdapterCore(AdapterConfig(sbr_root=sbr_root, **adapter_kwargs))
    original = ranking_agent_cls._rank_hypotheses

    async def _rank_hypotheses(self, hypotheses, criteria):  # type: ignore[no-untyped-def]
        run_id = f"{run_id_prefix}_{getattr(self, 'agent_id', 'ranking')}"
        try:
            return build_rankings(
                hypotheses=list(hypotheses),
                core=core,
                campaign_context=campaign_context,
                run_id=run_id,
            )
        except Exception:
            logger.exception("binder-adapter ranking failed; falling back to Jnana's ranking")
            return await original(self, hypotheses, criteria)

    ranking_agent_cls._rank_hypotheses = _rank_hypotheses

    def restore() -> None:
        ranking_agent_cls._rank_hypotheses = original

    return restore
