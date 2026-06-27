from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agent_memory.manager import Memory


@dataclass
class EvalCase:
    query: str
    expected_action: str
    notes: str = ""


@dataclass
class EvalDataset:
    name: str
    description: str = ""
    memories: list[dict] = field(default_factory=list)
    cases: list[EvalCase] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> EvalDataset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = [
            EvalCase(
                query=c["query"],
                expected_action=c["expected_action"],
                notes=c.get("notes", ""),
            )
            for c in data.get("cases", [])
        ]
        return cls(
            name=data.get("name", Path(path).stem),
            description=data.get("description", ""),
            memories=data.get("memories", []),
            cases=cases,
        )


@dataclass
class EvalResult:
    dataset: str
    total: int = 0
    correct: int = 0
    failures: list[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "total": self.total,
            "correct": self.correct,
            "precision": round(self.precision, 4),
            "failures": self.failures,
        }


def seed_dataset(memory: Memory, dataset: EvalDataset) -> None:
    for item in dataset.memories:
        memory.remember(
            query=item["query"],
            response=item["response"],
            type=item.get("type", "conversation"),
            scope=item.get("scope", "user"),
            tags=item.get("tags", []),
            ttl=item.get("ttl"),
        )


def run_eval(memory: Memory, dataset: EvalDataset) -> EvalResult:
    result = EvalResult(dataset=dataset.name)

    for case in dataset.cases:
        decision = memory.resolve(case.query)
        result.total += 1
        actual = decision.action.value

        # Allow flexible matching: restore accepts replay; verify accepts restore/replay
        acceptable = {case.expected_action}
        if case.expected_action == "restore":
            acceptable |= {"replay"}
        if case.expected_action == "replay":
            acceptable |= {"restore"}
        if case.expected_action == "verify":
            acceptable |= {"restore", "replay"}

        if actual in acceptable:
            result.correct += 1
        else:
            result.failures.append(
                {
                    "query": case.query,
                    "expected": case.expected_action,
                    "actual": actual,
                    "confidence": round(decision.confidence, 4),
                    "reasons": decision.reasons,
                }
            )

    return result


def run_eval_suite(memory: Memory, dataset_paths: list[Path]) -> list[EvalResult]:
    results: list[EvalResult] = []
    for path in dataset_paths:
        dataset = EvalDataset.load(path)
        seed_dataset(memory, dataset)
        results.append(run_eval(memory, dataset))
    return results


def format_eval_report(results: list[EvalResult]) -> str:
    lines = ["Agent Memory Evaluation", "=======================", ""]
    for result in results:
        lines.append(f"Dataset: {result.dataset}")
        lines.append(f"  Cases:     {result.total}")
        lines.append(f"  Correct:   {result.correct}")
        lines.append(f"  Precision: {result.precision:.1%}")
        if result.failures:
            lines.append(f"  Failures:  {len(result.failures)}")
        lines.append("")
    overall_total = sum(r.total for r in results)
    overall_correct = sum(r.correct for r in results)
    if overall_total:
        lines.append(f"Overall precision: {overall_correct / overall_total:.1%}")
    return "\n".join(lines)
