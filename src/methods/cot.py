"""Standard Chain-of-Thought baseline."""

from __future__ import annotations

import textwrap
import time

from ..evaluator import extract_answer
from ..models import BenchmarkProblem, ExperimentRow, MethodName

# ── Prompts ─────────────────────────────────────────────────────────

COT_SYSTEM = textwrap.dedent("""\
You are a precise problem-solving assistant.
Solve the problem step by step, showing your full reasoning.
After your reasoning, clearly state your final answer on the last line
in the format: "Final Answer: <answer>"
""")


# ── Method ──────────────────────────────────────────────────────────

class ChainOfThought:
    name = MethodName.COT

    def solve(self, problem: BenchmarkProblem, llm) -> ExperimentRow:
        start = time.time()
        tokens_before = llm.total_tokens_used

        try:
            raw = llm.call_raw(COT_SYSTEM, problem.text)
            predicted = extract_answer(raw, problem.benchmark)
        except Exception as exc:
            return ExperimentRow(
                benchmark=problem.benchmark.value,
                problem_id=problem.id,
                method=self.name.value,
                model=llm.model_name,
                ground_truth=problem.answer,
                error=str(exc),
            )

        from ..evaluator import check_answer
        correct = check_answer(predicted, problem.answer, problem.benchmark)

        return ExperimentRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            method=self.name.value,
            model=llm.model_name,
            predicted_answer=predicted,
            ground_truth=problem.answer,
            correct=correct,
            tokens_used=llm.total_tokens_used - tokens_before,
            latency_s=round(time.time() - start, 3),
        )
