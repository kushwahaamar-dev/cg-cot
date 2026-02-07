"""CG-CoT: Symmetric Code-Grounded Chain-of-Thought (Bidirectional)."""

from __future__ import annotations

import logging
import textwrap
import time

from ..code_engine import build_translation_prompt, compare_nl_and_code, execute_code
from ..config import MAX_REGEN_ATTEMPTS, MODEL_VERIFIER
from ..evaluator import extract_answer
from ..llm_client import make_client
from ..models import (
    BenchmarkProblem,
    CodeTranslation,
    ExperimentRow,
    MethodName,
    ReasoningChain,
    ReasoningStep,
)

logger = logging.getLogger(__name__)

# ── Prompts ─────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = textwrap.dedent("""\
You are a precise step-by-step reasoning engine.
Given a problem, solve it step by step. For each step, output:
  - step_id: integer starting at 1
  - nl_reasoning: your natural language reasoning for this step
  - conclusion: what this step concludes (a short statement or number)
  - is_codeable: true if this step involves a calculation, logical operation,
    or comparison that could be verified with Python code; false if it's
    purely commonsense or qualitative.

After all steps, provide your final_answer (just the answer, no explanation).
""")

DEBUG_CODE_SYSTEM = textwrap.dedent("""\
You are a Python debugging expert.
You wrote code to verify a reasoning step, but the code result disagrees with the reasoning
or failed to execute.
Check if the code has a bug (e.g., integer division, wrong formula, syntax error).
If the reasoning seems correct and the code is wrong, FIX the code.
If the code is correct and the reasoning is wrong, keep the code as is.
""")

DEBUG_CODE_USER = textwrap.dedent("""\
Reasoning Step: "{nl_reasoning}"
Expected Conclusion: {conclusion}

Your Code:
{code}

Execution Output:
{output}

Error (if any):
{error}

Please fix the python_code if it is buggy. Return the fixed CodeTranslation.
""")

REGEN_SYSTEM = textwrap.dedent("""\
You are a careful reasoning assistant correcting a potential error.
The code verification channel (after debugging) disagrees with your conclusion.
Please re-examine the step and correct it if needed.

Output the corrected step with the same JSON schema:
  - step_id, nl_reasoning, conclusion, is_codeable
""")

REGEN_USER = textwrap.dedent("""\
Original problem context:
{context}

Your previous reasoning for step {step_id}:
  "{nl_reasoning}"
  Conclusion: {old_conclusion}

Verified Code Result: {code_answer}

Please reconsider this step. If the code result is more trustworthy,
update your conclusion accordingly. Output the corrected step.
""")


# ── Method ──────────────────────────────────────────────────────────

class SymmetricCGCoT:
    name = MethodName.SYMMETRIC_CGCOT

    def __init__(self):
        # Secondary client for the "Code/Verifier" role
        self.verifier_client = make_client(provider="ollama", model=MODEL_VERIFIER)

    def solve(self, problem: BenchmarkProblem, reasoner_client) -> ExperimentRow:
        start = time.time()
        tokens_before = reasoner_client.total_tokens_used + self.verifier_client.total_tokens_used

        steps_total = 0
        steps_verified = 0
        steps_disagreed = 0
        steps_recovered = 0
        steps_crashed = 0
        
        # Track recovery types
        # recovered_by_code_fix: Logic was right, Code was buggy -> fixed code
        # recovered_by_logic_fix: Code was right, Logic was buggy -> fixed logic

        try:
            # ── 1. Decompose (Reasoner) ─────────────────────────────
            chain = reasoner_client.call(DECOMPOSE_SYSTEM, problem.text, ReasoningChain)
            steps_total = len(chain.steps)

            # ── 2. Verify each codeable step ────────────────────────
            for i, step in enumerate(chain.steps):
                if not step.is_codeable:
                    continue

                # 2a. Translate (Verifier)
                sys_prompt, usr_prompt = build_translation_prompt(step)
                try:
                    translation = self.verifier_client.call(sys_prompt, usr_prompt, CodeTranslation)
                except Exception:
                    logger.debug("Code translation failed for step %d", step.step_id)
                    continue

                # 2b. Execute
                code_result = execute_code(translation.python_code)
                
                # Monitor crashes
                if not code_result.executed:
                    steps_crashed += 1

                # 2c. Compare
                code_result = compare_nl_and_code(step.conclusion, code_result)
                
                # If Agree: Done
                if code_result.agrees_with_nl is True:
                    steps_verified += 1
                    continue
                
                # If Disagree/Unknown:
                steps_disagreed += 1
                logger.debug(
                        "Disagreement/Crash at step %d: NL='%s' CodeOut='%s' Err='%s'",
                        step.step_id, step.conclusion, code_result.stdout, code_result.stderr
                    )

                # ── 3. Bidirectional Loop ───────────────────────────
                
                # Branch A: Debug Code (Verifier)
                # "Logic acts as Supervisor"
                try:
                    debug_prompt = DEBUG_CODE_USER.format(
                        nl_reasoning=step.nl_reasoning,
                        conclusion=step.conclusion,
                        code=translation.python_code,
                        output=code_result.stdout,
                        error=code_result.stderr
                    )
                    fixed_trans = self.verifier_client.call(DEBUG_CODE_SYSTEM, debug_prompt, CodeTranslation)
                    
                    # Execute Fixed Code
                    fixed_result = execute_code(fixed_trans.python_code)
                    fixed_result = compare_nl_and_code(step.conclusion, fixed_result)
                    
                    if fixed_result.agrees_with_nl is True:
                        # Success! Logic was correct, Code was buggy.
                        steps_recovered += 1
                        logger.debug(
                            "Step %d recovered by FIXING CODE.\nOLD Code: %s\nNEW Code: %s\nOutput: %s",
                            step.step_id, translation.python_code, fixed_trans.python_code, fixed_result.stdout
                        )
                        continue # Trust the original logic
                    
                    # If still disagrees/crashes, assume Code might be right (or Logic is very wrong)
                    # Fall through to Branch B
                    if fixed_result.executed and fixed_result.stdout:
                         # Use the fixed code result for feedback if available
                         code_result = fixed_result

                except Exception as e:
                    logger.debug("Debug code branch failed: %s", e)

                # Branch B: Regenerate Logic (Reasoner)
                # "Code acts as Supervisor"
                if not code_result.executed or not code_result.stdout:
                    # If code is still broken, we can't offer feedback.
                    continue

                for _attempt in range(MAX_REGEN_ATTEMPTS):
                    regen_user = REGEN_USER.format(
                        context=problem.text[:500],
                        step_id=step.step_id,
                        nl_reasoning=step.nl_reasoning,
                        old_conclusion=step.conclusion,
                        code_answer=code_result.code_answer,
                    )
                    try:
                        corrected = reasoner_client.call(REGEN_SYSTEM, regen_user, ReasoningStep)
                        # Check agreement again? Or just accept?
                        # Standard CG-CoT accepts the correction.
                        chain.steps[i] = corrected
                        steps_recovered += 1
                        logger.debug(
                            "Step %d recovered by FIXING LOGIC: '%s' → '%s'",
                            step.step_id, step.conclusion, corrected.conclusion,
                        )
                        logger.debug("Corrected Reasoning: %s", corrected.nl_reasoning)
                        break
                    except Exception:
                        pass

            # ── 4. Extract final answer ─────────────────────────────
            if steps_recovered > 0:
                # Re-derive from corrected chain
                corrected_reasoning = "\n".join(
                    f"Step {s.step_id}: {s.conclusion}" for s in chain.steps
                )
                rederive_prompt = (
                    f"Based on these corrected reasoning steps:\n{corrected_reasoning}\n\n"
                    f"Original problem: {problem.text[:300]}\n\n"
                    "What is the final answer? Respond with ONLY the answer."
                )
                final_raw = reasoner_client.call_raw(
                    "Extract the final answer from the reasoning steps.", rederive_prompt
                )
                predicted = extract_answer(final_raw, problem.benchmark)
            else:
                predicted = extract_answer(chain.final_answer, problem.benchmark)

        except Exception as exc:
            return ExperimentRow(
                benchmark=problem.benchmark.value,
                problem_id=problem.id,
                method=self.name.value,
                model=reasoner_client.model_name,
                ground_truth=problem.answer,
                steps_total=steps_total,
                steps_verified=steps_verified,
                steps_disagreed=steps_disagreed,
                steps_recovered=steps_recovered,
                steps_crashed=steps_crashed,
                error=str(exc),
            )

        from ..evaluator import check_answer
        correct = check_answer(predicted, problem.answer, problem.benchmark)
        
        total_tokens = (reasoner_client.total_tokens_used + self.verifier_client.total_tokens_used) - tokens_before

        return ExperimentRow(
            benchmark=problem.benchmark.value,
            problem_id=problem.id,
            method=self.name.value,
            model=reasoner_client.model_name,
            predicted_answer=predicted,
            ground_truth=problem.answer,
            correct=correct,
            tokens_used=total_tokens,
            latency_s=round(time.time() - start, 3),
            steps_total=steps_total,
            steps_verified=steps_verified,
            steps_disagreed=steps_disagreed,
            steps_recovered=steps_recovered,
            steps_crashed=steps_crashed,
        )
