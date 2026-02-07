"""Program-of-Thought baseline: solve entirely via code execution."""

from __future__ import annotations

import textwrap
import time

from ..code_engine import execute_code
from ..evaluator import extract_answer
from ..models import BenchmarkProblem, CodeTranslation, ExperimentRow, MethodName

# ── Prompts ─────────────────────────────────────────────────────────

POT_SYSTEM = textwrap.dedent("""\
You are a code-based problem solver. Convert the problem into a Python
program that computes the answer. The program must:
1. Use only the Python standard library.
2. Print ONLY the final numeric or textual answer on the last line.
3. Do NOT import os, sys, subprocess, or any I/O libraries.
""")


# ── Method ──────────────────────────────────────────────────────────

class ProgramOfThought:
    name = MethodName.POT

    def solve(self, problem: BenchmarkProblem, llm) -> ExperimentRow:
        start = time.time()
        tokens_before = llm.total_tokens_used

        try:
            # Ask LLM to write Python code
            translation = llm.call(
                POT_SYSTEM, problem.text, CodeTranslation,
            )
            code = translation.python_code

            # Execute
            result = execute_code(code)

            if result.executed and result.stdout:
                predicted = extract_answer(result.stdout, problem.benchmark)
            else:
                # Code failed → fall back to raw LLM answer
                predicted = ""

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
