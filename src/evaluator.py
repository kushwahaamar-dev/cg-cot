"""Answer extraction and correctness verification per benchmark."""

from __future__ import annotations

import logging
import re

from .models import BenchmarkName

logger = logging.getLogger(__name__)


# ── Extraction helpers ──────────────────────────────────────────────

def _extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return text[start:].strip()


def _extract_number(text: str) -> str | None:
    """Find the last standalone number (possibly negative, decimal) in text."""
    # First try #### pattern (GSM8K style)
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    # Then try boxed pattern (MATH style)
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed
    # Last number in text
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _extract_label(text: str) -> str | None:
    """Extract True/False/Unknown label from text."""
    t = text.lower().strip()
    for label in ("true", "false", "unknown"):
        if label in t:
            return label.capitalize()
    return None


def _normalize_math_answer(s: str) -> str:
    """Light normalization for MATH answers."""
    s = s.strip()
    s = s.replace("\\$", "").replace("$", "")
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\ ", " ")
    # Strip \text{...} wrapper but keep content
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "")
    s = s.replace("\\dfrac", "\\frac")
    s = s.strip()
    return s


def _try_numeric_eq(a: str, b: str) -> bool | None:
    """Compare two strings as floats if possible."""
    try:
        fa = float(a.replace(",", ""))
        fb = float(b.replace(",", ""))
        return abs(fa - fb) < 1e-6
    except (ValueError, TypeError):
        return None


def _try_sympy_eq(a: str, b: str) -> bool | None:
    """Compare via SymPy parse + simplify (for equivalent math expressions)."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify, Rational

        ea = parse_latex(a)
        eb = parse_latex(b)
        return simplify(ea - eb) == 0
    except Exception:
        return None


# ── Public API ──────────────────────────────────────────────────────

def extract_answer(raw_response: str, benchmark: BenchmarkName) -> str:
    """Pull the predicted answer out of a model's raw response."""
    if benchmark == BenchmarkName.GSM8K:
        num = _extract_number(raw_response)
        return num if num is not None else raw_response.strip()

    if benchmark == BenchmarkName.MATH:
        # Try boxed first (handles nested braces), then last number
        boxed = _extract_boxed(raw_response)
        if boxed is not None:
            return _normalize_math_answer(boxed)
        num = _extract_number(raw_response)
        return num if num is not None else _normalize_math_answer(raw_response)

    if benchmark == BenchmarkName.FOLIO:
        label = _extract_label(raw_response)
        return label if label is not None else raw_response.strip()

    return raw_response.strip()


def check_answer(predicted: str, ground_truth: str, benchmark: BenchmarkName) -> bool:
    """Return True if *predicted* matches *ground_truth* for the benchmark."""
    pred = predicted.strip()
    gt = ground_truth.strip()

    if not pred or not gt:
        return False

    # ── GSM8K: numeric equality ─────────────────────────────────
    if benchmark == BenchmarkName.GSM8K:
        result = _try_numeric_eq(pred, gt)
        return result if result is not None else pred == gt

    # ── MATH: multi-strategy comparison ─────────────────────────
    if benchmark == BenchmarkName.MATH:
        pred_n = _normalize_math_answer(pred)
        gt_n = _normalize_math_answer(gt)
        # exact string
        if pred_n == gt_n:
            return True
        # numeric
        result = _try_numeric_eq(pred_n, gt_n)
        if result is not None:
            return result
        # SymPy
        result = _try_sympy_eq(pred_n, gt_n)
        if result is not None:
            return result
        return False

    # ── FOLIO: label match ──────────────────────────────────────
    if benchmark == BenchmarkName.FOLIO:
        return pred.lower().strip() == gt.lower().strip()

    return pred == gt
