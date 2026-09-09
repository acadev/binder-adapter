"""Verify the Jnana ranking hook against the REAL RankingAgent class.

Skips when the Jnana checkout is unavailable. These tests import Jnana's actual
`RankingAgent` and `ResearchHypothesis`, patch the ranking method, and drive it
through an asyncio event loop the way ProtoGnosis does.
"""

from __future__ import annotations

import asyncio
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

JNANA_ROOT = os.environ.get("BINDER_ADAPTER_JNANA_ROOT", "/Users/ramanathana/Work/Jnana")
SBR_ROOT = os.environ.get("BINDER_ADAPTER_SBR_ROOT", "/Users/ramanathana/Work/StructBioReasoner")

FW = "MSTGEELQKAWDIVKRTGDKLYFRNPETGKWEWVQ"
AA = "ACDEFGHIKLMNPQRSTVWY"


def _probe(root: str, snippet: str) -> bool:
    if not Path(root).is_dir():
        return False
    code = f"import sys; sys.path.insert(0, r'{root}'); {snippet}; print('ok')"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=240)
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False


_JNANA_OK = _probe(JNANA_ROOT, "from jnana.protognosis.agents.ranking_agent import RankingAgent")
_SBR_OK = _probe(
    SBR_ROOT,
    "from struct_bio_reasoner.agents.computational_design.quality_control import SequenceQualityControl",
)

requires_both = pytest.mark.skipif(
    not (_JNANA_OK and _SBR_OK),
    reason=f"need both checkouts importable (jnana={_JNANA_OK}, sbr={_SBR_OK})",
)

if _JNANA_OK and JNANA_ROOT not in sys.path:
    sys.path.insert(0, JNANA_ROOT)


def mutate(seq: str, n: int, seed: int) -> str:
    r = random.Random(seed)
    s = list(seq)
    for p in r.sample(range(len(seq)), n):
        s[p] = r.choice([a for a in AA if a != s[p]])
    return "".join(s)


@pytest.fixture
def hypotheses():
    """Real Jnana ResearchHypothesis objects."""
    from jnana.protognosis.core.agent_core import ResearchHypothesis  # type: ignore

    specs = [
        ("h_in_1", mutate(FW, 9, 101)),    # in budget
        ("h_in_2", mutate(FW, 14, 202)),   # in budget
        ("h_too_few", mutate(FW, 2, 303)), # below floor -> rejected
    ]
    return [
        ResearchHypothesis(
            content=f"Proposed binder sequence: {seq}",
            summary=f"binder variant {hid}",
            agent_id="generation-test",
            hypothesis_id=hid,
        )
        for hid, seq in specs
    ]


@requires_both
def test_patch_and_restore_roundtrip():
    from jnana.protognosis.agents.ranking_agent import RankingAgent  # type: ignore
    from binder_adapter.jnana_hook import patch_ranking_agent

    original = RankingAgent._rank_hypotheses
    restore = patch_ranking_agent(
        campaign_context={"framework_sequence": FW, "target_ids": ["T1"]},
        sbr_root=SBR_ROOT,
    )
    assert RankingAgent._rank_hypotheses is not original
    restore()
    assert RankingAgent._rank_hypotheses is original


@requires_both
def test_patched_ranking_uses_real_sbr_objectives(tmp_path, hypotheses):
    from jnana.protognosis.agents.ranking_agent import RankingAgent  # type: ignore
    from binder_adapter.jnana_hook import get_last_resample_request, patch_ranking_agent

    restore = patch_ranking_agent(
        campaign_context={"framework_sequence": FW, "target_ids": ["T1"]},
        sbr_root=SBR_ROOT,
        artifact_dir=str(tmp_path / "runs"),
        timeout_seconds=600,
    )
    try:
        agent = RankingAgent.__new__(RankingAgent)  # bypass LLM/memory wiring
        agent.agent_id = "ranking-test"

        rankings = asyncio.run(agent._rank_hypotheses(hypotheses, "overall_quality"))

        # Jnana's existing contract is preserved
        for r in rankings:
            assert {"hypothesis_id", "rank", "score", "justification"} <= set(r)

        # ...and enriched with real multi-objective detail
        assert rankings, "expected at least one ranked candidate"
        first = rankings[0]
        assert "developability" in first["active_objectives"]
        assert first["objective_reports"][0]["developability"]["available"] is True
        assert first["score"] is not None

        # the out-of-budget candidate was rejected, not ranked
        ranked_ids = {r["hypothesis_id"] for r in rankings}
        assert "h_too_few" not in ranked_ids

        resample = get_last_resample_request()
        assert resample["needs_resample"] is True
        assert "h_too_few" in resample["invalid_candidate_ids"]
    finally:
        restore()


@requires_both
def test_ranking_falls_back_when_adapter_raises(tmp_path, hypotheses):
    """A broken adapter must not take the tournament down."""
    from jnana.protognosis.agents.ranking_agent import RankingAgent  # type: ignore
    from binder_adapter.jnana_hook import patch_ranking_agent

    # missing framework_sequence -> extraction raises ValueError inside the hook
    restore = patch_ranking_agent(
        campaign_context={"target_ids": ["T1"]},
        sbr_root=SBR_ROOT,
        artifact_dir=str(tmp_path / "runs"),
    )
    try:
        agent = RankingAgent.__new__(RankingAgent)
        agent.agent_id = "ranking-fallback"

        called = {"n": 0}

        async def fake_llm_rank(self, hyps, criteria):
            called["n"] += 1
            return [{"hypothesis_id": h.hypothesis_id, "rank": i + 1} for i, h in enumerate(hyps)]

        # point the captured original at our probe by re-patching over it
        import binder_adapter.jnana_hook as hook

        restore()
        RankingAgent._rank_hypotheses = fake_llm_rank
        restore = hook.patch_ranking_agent(
            campaign_context={"target_ids": ["T1"]},  # still broken
            sbr_root=SBR_ROOT,
            artifact_dir=str(tmp_path / "runs2"),
        )

        rankings = asyncio.run(agent._rank_hypotheses(hypotheses, "overall_quality"))
        assert called["n"] == 1, "fallback to Jnana's own ranking should have fired"
        assert len(rankings) == len(hypotheses)
    finally:
        restore()
