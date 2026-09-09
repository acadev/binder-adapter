from __future__ import annotations

import re
from typing import Any, Optional

from .schemas import SCHEMA_VERSION, CandidateArtifact

AA_RUN = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{8,}")

# Words that are all-AA-letters but are obviously prose, not sequences.
_ENGLISH_DECOYS = {
    "AFFINITY", "STABILITY", "MATERIAL", "CANDIDATE", "SPECIFIC",
    "BINDING", "PROTEIN", "BINDER", "BATCH", "PREDICT",
}


def _looks_like_sequence(token: str) -> bool:
    if token in _ENGLISH_DECOYS:
        return False
    # Real designed sequences use a broad alphabet; prose runs do not.
    return len(set(token)) >= 6


def extract_sequence_from_text(text: str) -> tuple[Optional[str], list[str]]:
    """Pull a candidate sequence out of free text.

    Preference order:
      1. explicit `sequence:`/`binder:` label
      2. FASTA body
      3. longest plausible amino-acid run
    """
    errors: list[str] = []
    if not text:
        return None, ["EMPTY_CONTENT"]

    labelled = re.search(
        r"(?:binder[_ ]?sequence|candidate[_ ]?sequence|sequence|binder)\s*[:=]\s*([A-Za-z]{8,})",
        text,
        re.IGNORECASE,
    )
    if labelled:
        seq = labelled.group(1).upper()
        if AA_RUN.fullmatch(seq):
            return seq, errors
        errors.append("LABELLED_SEQUENCE_HAS_NON_AA_CHARS")

    if ">" in text:
        fasta_body = "".join(
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith(">")
        ).upper()
        if fasta_body and AA_RUN.fullmatch(fasta_body):
            return fasta_body, errors

    runs = [m.group(0) for m in AA_RUN.finditer(text.upper())]
    plausible = [r for r in runs if _looks_like_sequence(r)]
    if plausible:
        plausible.sort(key=len, reverse=True)
        return plausible[0], errors

    errors.append("SEQUENCE_EXTRACTION_FAILED")
    return None, errors


def extract_all_sequences(text: str) -> list[str]:
    """Return every plausible amino-acid run in the text (dedup, order preserved)."""
    if not text:
        return []
    seen, out = set(), []
    for m in AA_RUN.finditer(text.upper()):
        s = m.group(0)
        if _looks_like_sequence(s) and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def select_best_candidate_sequence(
    text: str,
    framework: str,
    min_frac: float = 0.20,
    max_frac: float = 0.50,
) -> tuple[Optional[str], list[str]]:
    """Framework-aware selection when the text mentions several sequences.

    Jnana hypotheses typically quote the wild-type framework AND one or more
    proposed variants (and a lot of English prose). This picks the sequence that
    best fits the mutation budget:

      1. an explicit ``candidate_sequence:``/``sequence:`` label wins if present
         and in budget;
      2. otherwise, among runs whose LENGTH is within +/-2 of the framework
         (real variants, not English words like "SPECIFICALLY"), prefer the one
         whose mutated_fraction is inside [min_frac, max_frac] and closest to the
         budget midpoint;
      3. if nothing qualifies, return None with a diagnostic error rather than a
         misleading decoy — honest failure beats a fake candidate.
    """
    errors: list[str] = []
    if not framework:
        return extract_sequence_from_text(text)

    L = len(framework)
    mid = (min_frac + max_frac) / 2.0

    def _score(seq: str):
        frac, _ = mutation_metrics(seq, framework)
        return frac

    # 1) explicit label
    labelled = re.findall(
        r"(?:binder[_ ]?sequence|candidate[_ ]?sequence|sequence)\s*[:=]\s*([ACDEFGHIKLMNPQRSTVWY]{8,})",
        text,
        re.IGNORECASE,
    )
    length_ok = [s.upper() for s in labelled if abs(len(s) - L) <= 2]
    labelled_in_budget = [(abs(_score(s) - mid), s) for s in length_ok if min_frac <= _score(s) <= max_frac]
    if labelled_in_budget:
        labelled_in_budget.sort(key=lambda t: t[0])
        return labelled_in_budget[0][1], errors

    # 2) any run whose length is close to the framework (filters English words)
    near_len = [s for s in extract_all_sequences(text) if abs(len(s) - L) <= 2]
    in_budget = [(abs(_score(s) - mid), s) for s in near_len if min_frac <= _score(s) <= max_frac]
    if in_budget:
        in_budget.sort(key=lambda t: t[0])
        return in_budget[0][1], errors

    # 3) honest failure
    if near_len:
        errors.append("NO_IN_BUDGET_SEQUENCE_FOUND")
    else:
        errors.append("NO_FRAMEWORK_LENGTH_SEQUENCE_FOUND")
    return None, errors


def mutation_metrics(candidate_sequence: str, framework_sequence: str) -> tuple[float, int]:
    """(fraction, count) of positions differing from the framework.

    Length differences count as mutations; the fraction is normalised by the
    framework length, per the agreed mutation-budget definition.
    """
    L = len(framework_sequence)
    if L == 0:
        return 1.0, 0
    if not candidate_sequence:
        return 1.0, L

    overlap = min(len(candidate_sequence), L)
    diffs = sum(1 for i in range(overlap) if candidate_sequence[i] != framework_sequence[i])
    diffs += abs(len(candidate_sequence) - L)
    return diffs / float(L), diffs


def extract_candidate_artifacts_from_jnana(
    hypotheses: list[dict[str, Any]],
    campaign_context: dict[str, Any],
    schema_version: str = SCHEMA_VERSION,
) -> list[CandidateArtifact]:
    """Build CandidateArtifacts from Jnana hypotheses.

    Jnana hypotheses carry free text today, so sequences are parsed
    heuristically and extraction_confidence reflects how they were obtained.
    Once Jnana emits structured candidates this becomes a straight marshal.
    """
    framework = campaign_context.get("framework_sequence")
    if not framework:
        raise ValueError("campaign_context['framework_sequence'] is required for the mutation budget")
    framework = str(framework).upper()

    targets = campaign_context.get("target_ids") or campaign_context.get("targets") or []
    if isinstance(targets, str):
        targets = [targets]

    antigen_ref = str(campaign_context.get("antigen_structure_ref", "") or "")
    default_pdb = str(campaign_context.get("complex_pdb_path", "") or "")
    # Use the campaign's actual mutation budget for framework-aware extraction so
    # we pick the variant the caller will accept (defaults match AdapterConfig).
    min_frac = float(campaign_context.get("min_mutated_fraction", 0.20))
    max_frac = float(campaign_context.get("max_mutated_fraction", 0.50))

    artifacts: list[CandidateArtifact] = []
    for hyp in hypotheses:
        cid = str(hyp.get("hypothesis_id") or hyp.get("candidate_id") or hyp.get("id") or "")

        # Structured field wins over prose when Jnana provides one.
        structured = hyp.get("candidate_sequence") or hyp.get("sequence")
        if structured:
            seq: Optional[str] = str(structured).upper()
            errors: list[str] = []
            confidence = 1.0
        else:
            # Framework-aware: prefer an in-budget variant over the wild-type the
            # hypothesis text usually quotes alongside its proposals.
            seq, errors = select_best_candidate_sequence(
                str(hyp.get("content") or ""), framework, min_frac, max_frac
            )
            confidence = 0.6 if seq else 0.0

        frac, count = mutation_metrics(seq or "", framework)

        artifacts.append(
            CandidateArtifact(
                schema_version=schema_version,
                candidate_id=cid,
                target_ids=[str(t) for t in targets],
                framework_sequence=framework,
                candidate_sequence=seq or "",
                mutation_fraction=frac,
                mutation_count=count,
                antigen_structure_ref=antigen_ref,
                complex_pdb_path=str(hyp.get("complex_pdb_path", "") or default_pdb),
                provenance={
                    "source": "jnana_hypothesis",
                    "extraction": "structured" if structured else "text_heuristic",
                    "jnana_agent_strategy": hyp.get("strategy") or hyp.get("agent_id") or "",
                },
                extraction_confidence=confidence,
                extraction_errors=errors,
            )
        )

    return artifacts
