"""Pydantic schemas for every structured data exchange in the pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator


# ── Enums ───────────────────────────────────────────────────────────

class BenchmarkName(str, Enum):
    GSM8K = "gsm8k"
    MATH = "math"
    FOLIO = "folio"


class MethodName(str, Enum):
    COT = "cot"
    POT = "pot"
    SC = "self_consistency"
    CGCOT = "cgcot"
    SYMMETRIC_CGCOT = "symmetric_cgcot"  # New bidirectional method


# ── Benchmark ───────────────────────────────────────────────────────

class BenchmarkProblem(BaseModel):
    """Unified problem representation across all benchmarks."""
    id: str
    benchmark: BenchmarkName
    text: str                              # full problem statement
    answer: str                            # ground-truth answer (string form)
    difficulty: Optional[str] = None       # e.g. MATH level, GSM8K step count
    metadata: dict = Field(default_factory=dict)


# ── Reasoning chain (LLM structured output) ─────────────────────────

class ReasoningStep(BaseModel):
    """A single step in a chain-of-thought."""
    step_id: int
    nl_reasoning: str                      # natural-language explanation
    conclusion: str                        # what this step concludes
    is_codeable: bool = True               # can this step be verified with code?

    @field_validator("conclusion", "nl_reasoning", mode="before")
    @classmethod
    def coerce_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v


class ReasoningChain(BaseModel):
    """Full chain-of-thought with extracted final answer."""
    steps: list[ReasoningStep]
    final_answer: str

    @field_validator("final_answer", mode="before")
    @classmethod
    def coerce_final(cls, v):
        return str(v) if not isinstance(v, str) else v


# ── Code channel ────────────────────────────────────────────────────

class CodeTranslation(BaseModel):
    """LLM-generated Python code that computes a step's conclusion."""
    python_code: str
    description: str = ""                  # what the code checks


class CodeResult(BaseModel):
    """Result of executing translated code in the sandbox."""
    executed: bool = False
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    agrees_with_nl: Optional[bool] = None  # None = unknown/crash, True/False = verified
    code_answer: str = ""                  # parsed output from code


# ── Experiment logging ──────────────────────────────────────────────

class ExperimentRow(BaseModel):
    """One row in the results CSV."""
    benchmark: str
    problem_id: str
    method: str
    model: str
    predicted_answer: str = ""
    ground_truth: str = ""
    correct: bool = False
    tokens_used: int = 0
    latency_s: float = 0.0
    # Diagnostics
    steps_total: int = 0
    steps_verified: int = 0                # code ran successfully
    steps_disagreed: int = 0               # NL ≠ code
    steps_recovered: int = 0               # regen fixed the disagreement
    steps_crashed: int = 0                 # code failed to execute (NEW)
    error: str = ""                        # any exception message
