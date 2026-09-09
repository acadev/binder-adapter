from __future__ import annotations

import math

from .schemas import CandidateArtifact, InvalidCandidate, ValidationResult


def mutation_count_bounds(
    framework_length: int,
    min_mutated_fraction: float = 0.20,
    max_mutated_fraction: float = 0.50,
) -> tuple[int, int]:
    """Derive absolute mutation-count bounds from the framework length."""
    min_count = math.ceil(min_mutated_fraction * framework_length)
    max_count = math.floor(max_mutated_fraction * framework_length)
    return min_count, max_count


def validate_mutation_constraints(
    candidate: CandidateArtifact,
    framework_sequence: str,
    min_mutated_fraction: float = 0.20,
    max_mutated_fraction: float = 0.50,
) -> ValidationResult:
    """Hard mutation-budget gate. Reject-and-resample policy: never repair.

    Enforces BOTH the percent-of-positions constraint and the derived
    absolute mutation-count constraint.
    """
    reasons: list[str] = []

    L = len(framework_sequence)
    if L <= 0:
        reasons.append("FRAMEWORK_SEQUENCE_EMPTY")
        return ValidationResult(
            valid=False,
            invalid=InvalidCandidate(
                candidate_id=candidate.candidate_id,
                target_ids=list(candidate.target_ids),
                constraints_tripped=reasons,
                mutated_fraction=candidate.mutation_fraction,
                mutated_count=candidate.mutation_count,
                reason="framework sequence is empty; cannot evaluate mutation budget",
            ),
        )

    if candidate.extraction_errors:
        reasons.extend(candidate.extraction_errors)

    min_count, max_count = mutation_count_bounds(L, min_mutated_fraction, max_mutated_fraction)

    mf = candidate.mutation_fraction
    if mf < min_mutated_fraction or mf > max_mutated_fraction:
        reasons.append(
            f"MUTATION_FRACTION_OUT_OF_RANGE:{mf:.4f}"
            f" not in [{min_mutated_fraction},{max_mutated_fraction}]"
        )

    mc = candidate.mutation_count
    if mc < min_count or mc > max_count:
        reasons.append(f"MUTATION_COUNT_OUT_OF_RANGE:{mc} not in [{min_count},{max_count}]")

    if reasons:
        return ValidationResult(
            valid=False,
            invalid=InvalidCandidate(
                candidate_id=candidate.candidate_id,
                target_ids=list(candidate.target_ids),
                constraints_tripped=reasons,
                mutated_fraction=mf,
                mutated_count=mc,
                reason="; ".join(reasons),
            ),
        )

    return ValidationResult(valid=True, invalid=None)
