"""Self-Consistency baseline: sample k CoT paths, majority-vote the answer."""

from __future__ import annotations

import textwrap
import time
from collections import Counter

from ..config import SC_K, SC_TEMPERATURE
from ..evaluator import extract_answer
from ..models import BenchmarkProblem, ExperimentRow, MethodName

# ── Prompts (same as CoT) ──────────────────────────────────────────

SC_SYSTEM = textwrap.dedent("""\
You are a precise problem-solving assistant.
Solve the problem step by step, showing your full reasoning.
After your reasoning, clearly state your final answer on the last line
in the format: "Final Answer: <answer>"
""")


# ── Method ──────────────────────────────────────────────────────────

class SelfConsistency:
    name = MethodName.SC

    def __init__(self, k: int = SC_K):
        self.k = k

    def solve(self, problem: BenchmarkProblem, llm) -> ExperimentRow:
        start = time.time()
        tokens_before = llm.total_tokens_used

        # Temporarily raise temperature for diverse sampling
        original_temp = llm.temperature
        llm.temperature = SC_TEMPERATURE

        answers: list[str] = []
        try:
            for _ in range(self.k):
                raw = llm.call_raw(SC_SYSTEM, problem.text)
                ans = extract_answer(raw, problem.benchmark)
                answers.append(ans)
        except Exception as exc:
            llm.temperature = original_temp
            return ExperimentRow(
                benchmark=problem.benchmark.value,
                problem_id=problem.id,
                method=self.name.value,
                model=llm.model_name,
                ground_truth=problem.answer,
                error=str(exc),
            )
        finally:
            llm.temperature = original_temp

        # Majority vote
        if answers:
            counter = Counter(answers)
            predicted = counter.most_common(1)[0][0]
        else:
            predicted = ""

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
