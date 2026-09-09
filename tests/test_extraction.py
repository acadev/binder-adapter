from __future__ import annotations

from binder_adapter.extraction import (
    extract_candidate_artifacts_from_jnana,
    extract_sequence_from_text,
    mutation_metrics,
)

FW = "MSTGEELQKAWDIVKRTGDKLYFRNPETGKWEWVQ"


def test_labelled_sequence_wins():
    seq, errs = extract_sequence_from_text("Proposed binder sequence: MSTGEELQKAWDIVKRT  and notes")
    assert seq == "MSTGEELQKAWDIVKRT"
    assert errs == []


def test_fasta_body_is_parsed():
    seq, errs = extract_sequence_from_text(">design_1\nMSTGEELQKAWD\nIVKRTGDKLYFR\n")
    assert seq == "MSTGEELQKAWDIVKRTGDKLYFR"


def test_prose_without_sequence_reports_failure():
    seq, errs = extract_sequence_from_text("We should repack the hydrophobic core to improve binding.")
    assert seq is None
    assert "SEQUENCE_EXTRACTION_FAILED" in errs


def test_empty_content_reports_failure():
    seq, errs = extract_sequence_from_text("")
    assert seq is None
    assert "EMPTY_CONTENT" in errs


def test_mutation_metrics_counts_substitutions():
    cand = "M" + FW[1:]  # identical
    frac, count = mutation_metrics(cand, FW)
    assert count == 0 and frac == 0.0

    cand2 = "W" + FW[1:]  # one substitution
    frac2, count2 = mutation_metrics(cand2, FW)
    assert count2 == 1
    assert abs(frac2 - 1 / len(FW)) < 1e-9


def test_mutation_metrics_counts_length_difference():
    frac, count = mutation_metrics(FW + "AAA", FW)
    assert count == 3


def test_empty_candidate_is_fully_mutated():
    frac, count = mutation_metrics("", FW)
    assert frac == 1.0 and count == len(FW)


def test_structured_field_beats_text_heuristic():
    hyps = [{"hypothesis_id": "h1", "candidate_sequence": "MSTGEELQKA", "content": "ignore MMMMMMMMMMMMMMM"}]
    arts = extract_candidate_artifacts_from_jnana(hyps, {"framework_sequence": FW, "target_ids": ["T"]})
    assert arts[0].candidate_sequence == "MSTGEELQKA"
    assert arts[0].extraction_confidence == 1.0
    assert arts[0].provenance["extraction"] == "structured"


def test_missing_framework_raises():
    try:
        extract_candidate_artifacts_from_jnana([], {"target_ids": ["T"]})
    except ValueError as exc:
        assert "framework_sequence" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_targets_and_pdb_flow_from_context():
    hyps = [{"hypothesis_id": "h1", "content": "sequence: MSTGEELQKAWDIVKRT"}]
    ctx = {
        "framework_sequence": FW,
        "target_ids": ["T1", "T2"],
        "complex_pdb_path": "/tmp/x.pdb",
        "antigen_structure_ref": "ref.pdb",
    }
    art = extract_candidate_artifacts_from_jnana(hyps, ctx)[0]
    assert art.target_ids == ["T1", "T2"]
    assert art.complex_pdb_path == "/tmp/x.pdb"
    assert art.antigen_structure_ref == "ref.pdb"
