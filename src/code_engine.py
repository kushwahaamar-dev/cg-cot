"""NL→Python translator, sandboxed executor, and NL-vs-code comparator."""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

from .config import CODE_EXEC_TIMEOUT_S
from .models import CodeResult, ReasoningStep

logger = logging.getLogger(__name__)

# ── Blocked imports (safety) ────────────────────────────────────────

_BLOCKED_IMPORTS = re.compile(
    r"\b(import\s+(os|sys|shutil|subprocess|socket|http|urllib|pathlib|signal|ctypes)"
    r"|from\s+(os|sys|shutil|subprocess|socket|http|urllib|pathlib|signal|ctypes)\s+import)",
    re.IGNORECASE,
)


# ── Prompt for code translation ─────────────────────────────────────

CODE_TRANSLATE_SYSTEM = textwrap.dedent("""\
You are a code-grounding assistant. Your job is to translate a natural language
reasoning step into a short Python snippet that COMPUTES the result independently.

RULES:
1. Translate the PREMISES and OPERATIONS — do NOT hardcode the conclusion.
2. The code must PRINT exactly one value on the last line: the computed answer.
3. Use only the Python standard library (math, fractions, itertools, etc.) and sympy.
4. Do NOT import os, sys, subprocess, or any I/O libraries.
5. Keep the code under 20 lines.
""")

CODE_TRANSLATE_USER = textwrap.dedent("""\
Reasoning step:
{nl_reasoning}

Conclusion claimed: {conclusion}

Write a Python snippet that computes the result from the premises. The last
line must be a print() statement with the computed answer.
""")


def build_translation_prompt(step: ReasoningStep) -> tuple[str, str]:
    """Return (system, user) prompts for code translation."""
    user = CODE_TRANSLATE_USER.format(
        nl_reasoning=step.nl_reasoning,
        conclusion=step.conclusion,
    )
    return CODE_TRANSLATE_SYSTEM, user


# ── Sandboxed executor ──────────────────────────────────────────────

def execute_code(code: str, timeout: int = CODE_EXEC_TIMEOUT_S) -> CodeResult:
    """Run *code* in an isolated subprocess, return captured output."""
    # Safety: reject dangerous imports
    if _BLOCKED_IMPORTS.search(code):
        return CodeResult(
            executed=False,
            stderr="Blocked: code contains forbidden imports.",
        )

    try:
        # Write to a temp file to avoid shell-escaping issues
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            # Prepend standard imports to avoid trivial failures
            f.write("import math\nimport sympy\n" + code)
            tmp_path = f.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            # Inherit env so we can find installed packages like sympy
        )
        Path(tmp_path).unlink(missing_ok=True)

        is_success = (result.returncode == 0)
        return CodeResult(
            executed=is_success,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        Path(tmp_path).unlink(missing_ok=True)
        return CodeResult(executed=False, timed_out=True, stderr="Timed out.")
    except Exception as exc:
        return CodeResult(executed=False, stderr=str(exc))


# ── Comparator ──────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Lowercase, strip whitespace, remove trailing periods."""
    return s.lower().strip().rstrip(".")


def _try_float(s: str) -> float | None:
    """Parse a string as float, return None on failure."""
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def _try_sympy_eq(a: str, b: str) -> bool | None:
    """Check symbolic equality via SymPy (e.g. 1/2 == 0.5)."""
    try:
        from sympy import simplify, sympify
        from sympy.parsing.latex import parse_latex # specific for latex
        # robust parse
        sa = sympify(a, evaluate=False)
        sb = sympify(b, evaluate=False)
        return simplify(sa - sb) == 0
    except Exception:
        return None


def compare_nl_and_code(
    nl_conclusion: str,
    code_result: CodeResult,
) -> CodeResult:
    """Decide whether the NL conclusion agrees with the code output.

    Mutates and returns *code_result* with `agrees_with_nl` (True/False/None).
    """
    if not code_result.executed or not code_result.stdout:
        # Execution failed or no output -> UNKNOWN (None)
        # CRITICAL FIX: Do NOT return True here.
        code_result.agrees_with_nl = None 
        return code_result

    code_out = code_result.stdout.splitlines()[-1].strip()
    code_result.code_answer = code_out

    nl_norm = _normalize(nl_conclusion)
    code_norm = _normalize(code_out)

    # Exact match
    if nl_norm == code_norm:
        code_result.agrees_with_nl = True
        return code_result

    # Numeric comparison (tolerant)
    nl_f = _try_float(nl_norm)
    code_f = _try_float(code_norm)
    if nl_f is not None and code_f is not None:
        code_result.agrees_with_nl = abs(nl_f - code_f) < 1e-6
        return code_result
    
    # SymPy symbolic comparison
    # Try sympify on raw strings
    is_eq = _try_sympy_eq(nl_norm, code_norm)
    if is_eq is True:
        code_result.agrees_with_nl = True
        return code_result

    # Boolean / label comparison
    bool_map = {"true": "true", "false": "false", "yes": "true", "no": "false",
                "unknown": "unknown", "uncertain": "unknown"}
    nl_bool = bool_map.get(nl_norm)
    code_bool = bool_map.get(code_norm)
    if nl_bool and code_bool:
        code_result.agrees_with_nl = nl_bool == code_bool
        return code_result

    # Containment check (e.g. code prints "The answer is 42", NL says "42")
    if nl_norm in code_norm or code_norm in nl_norm:
        code_result.agrees_with_nl = True
        return code_result

    code_result.agrees_with_nl = False
    return code_result
