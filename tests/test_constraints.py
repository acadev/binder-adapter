from __future__ import annotations

import math

import pytest

from binder_adapter.constraints import mutation_count_bounds, validate_mutation_constraints
from binder_adapter.schemas import SCHEMA_VERSION, CandidateArtifact


def make_candidate(frac: float, count: int, framework: str, **kw) -> CandidateArtifact:
    return CandidateArtifact(
        schema_version=SCHEMA_VERSION,
        candidate_id=kw.pop("candidate_id", "c"),
        target_ids=["t1"],
        framework_sequence=framework,
        candidate_sequence=kw.pop("candidate_sequence", "X" * len(framework)),
        mutation_fraction=frac,
        mutation_count=count,
        antigen_structure_ref="",
        **kw,
    )


def test_count_bounds_are_ceil_and_floor():
    # L=35 -> ceil(0.2*35)=7, floor(0.5*35)=17
    assert mutation_count_bounds(35) == (7, 17)
    # L=100 -> exact
    assert mutation_count_bounds(100) == (20, 50)
    # L=33 -> ceil(6.6)=7, floor(16.5)=16
    assert mutation_count_bounds(33) == (7, 16)


@pytest.mark.parametrize("frac,count,valid", [
    (0.20, 20, True),    # lower boundary inclusive
    (0.50, 50, True),    # upper boundary inclusive
    (0.35, 35, True),    # mid-range
    (0.1999, 19, False), # just below floor
    (0.5001, 51, False), # just above ceiling
    (0.0, 0, False),
    (1.0, 100, False),
])
def test_fraction_boundaries(frac, count, valid):
    fw = "A" * 100
    res = validate_mutation_constraints(make_candidate(frac, count, fw), fw)
    assert res.valid is valid
    if not valid:
        assert res.invalid is not None
        assert res.invalid.constraints_tripped


def test_count_and_fraction_are_both_enforced():
    """A candidate can satisfy the fraction but violate the derived count."""
    fw = "A" * 35  # bounds are [7, 17]
    # fraction 0.20 is in range, but count 6 is below the derived floor of 7
    res = validate_mutation_constraints(make_candidate(0.20, 6, fw), fw)
    assert res.valid is False
    assert any("MUTATION_COUNT_OUT_OF_RANGE" in c for c in res.invalid.constraints_tripped)


def test_extraction_errors_propagate_as_rejection():
    fw = "A" * 100
    c = make_candidate(0.30, 30, fw, extraction_errors=["SEQUENCE_EXTRACTION_FAILED"])
    res = validate_mutation_constraints(c, fw)
    assert res.valid is False
    assert "SEQUENCE_EXTRACTION_FAILED" in res.invalid.constraints_tripped


def test_empty_framework_is_rejected_not_crashed():
    c = make_candidate(0.3, 3, "")
    res = validate_mutation_constraints(c, "")
    assert res.valid is False
    assert "FRAMEWORK_SEQUENCE_EMPTY" in res.invalid.constraints_tripped
