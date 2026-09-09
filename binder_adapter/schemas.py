from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "0.2"

OBJECTIVE_NAMES = [
    "binding_affinity",
    "specificity_off_target_proxy",
    "developability",
    "thermostability",
    "structure_confidence",
]


@dataclass(frozen=True)
class CandidateArtifact:
    schema_version: str
    candidate_id: str
    target_ids: list[str]
    framework_sequence: str
    candidate_sequence: str
    mutation_fraction: float
    mutation_count: int
    antigen_structure_ref: str = ""
    complex_pdb_path: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    extraction_confidence: float = 1.0
    extraction_errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectiveValue:
    """One objective for one candidate against one target.

    available=False means the objective could NOT be computed with the tooling
    present in this environment. score is None in that case and the objective is
    excluded from Pareto comparison. It is never back-filled with a guess.
    """

    score: Optional[float] = None
    uncertainty: Optional[float] = None
    maximize: bool = True
    available: bool = False
    unavailable_reason: str = ""

    @classmethod
    def unavailable(cls, reason: str) -> "ObjectiveValue":
        return cls(score=None, available=False, unavailable_reason=reason)

    @classmethod
    def measured(cls, score: float, uncertainty: Optional[float] = None) -> "ObjectiveValue":
        return cls(score=float(score), uncertainty=uncertainty, available=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "uncertainty": self.uncertainty,
            "maximize": self.maximize,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectiveValue":
        return cls(
            score=d.get("score"),
            uncertainty=d.get("uncertainty"),
            maximize=bool(d.get("maximize", True)),
            available=bool(d.get("available", False)),
            unavailable_reason=str(d.get("unavailable_reason", "")),
        )


@dataclass(frozen=True)
class ObjectiveReport:
    schema_version: str
    report_id: str
    candidate_id: str
    target_id: str
    objective_version: str
    binding_affinity: ObjectiveValue
    specificity_off_target_proxy: ObjectiveValue
    developability: ObjectiveValue
    thermostability: ObjectiveValue
    structure_confidence: ObjectiveValue
    constraints_tripped: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def value(self, objective_name: str) -> ObjectiveValue:
        return getattr(self, objective_name)

    def available_objectives(self) -> list[str]:
        return [o for o in OBJECTIVE_NAMES if self.value(o).available]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "candidate_id": self.candidate_id,
            "target_id": self.target_id,
            "objective_version": self.objective_version,
            **{o: self.value(o).to_dict() for o in OBJECTIVE_NAMES},
            "constraints_tripped": list(self.constraints_tripped),
            "evidence": list(self.evidence),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ObjectiveReport":
        return cls(
            schema_version=str(d.get("schema_version", SCHEMA_VERSION)),
            report_id=str(d["report_id"]),
            candidate_id=str(d["candidate_id"]),
            target_id=str(d["target_id"]),
            objective_version=str(d.get("objective_version", "unknown")),
            binding_affinity=ObjectiveValue.from_dict(d["binding_affinity"]),
            specificity_off_target_proxy=ObjectiveValue.from_dict(d["specificity_off_target_proxy"]),
            developability=ObjectiveValue.from_dict(d["developability"]),
            thermostability=ObjectiveValue.from_dict(d["thermostability"]),
            structure_confidence=ObjectiveValue.from_dict(d["structure_confidence"]),
            constraints_tripped=list(d.get("constraints_tripped", [])),
            evidence=list(d.get("evidence", [])),
        )


@dataclass(frozen=True)
class RankingDecision:
    candidate_id: str
    target_id: str
    rank: int
    pareto_front_id: int
    diversity_group_id: int
    score_summary_utility: Optional[float]
    justification: str
    active_objectives: list[str]
    objective_reports: list[ObjectiveReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_id": self.target_id,
            "rank": self.rank,
            "pareto_front_id": self.pareto_front_id,
            "diversity_group_id": self.diversity_group_id,
            "score_summary_utility": self.score_summary_utility,
            "justification": self.justification,
            "active_objectives": list(self.active_objectives),
            "objective_reports": [r.to_dict() for r in self.objective_reports],
        }


@dataclass(frozen=True)
class InvalidCandidate:
    candidate_id: str
    target_ids: list[str]
    constraints_tripped: list[str]
    mutated_fraction: float
    mutated_count: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "target_ids": list(self.target_ids),
            "constraints_tripped": list(self.constraints_tripped),
            "mutated_fraction": self.mutated_fraction,
            "mutated_count": self.mutated_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    invalid: Optional[InvalidCandidate] = None
