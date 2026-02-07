#!/usr/bin/env python3
"""Auto-populate the paper's tables and abstract with actual experiment results.

Usage:
    python populate_paper.py
"""

import re
from pathlib import Path

import pandas as pd

from src.analysis import (
    compute_accuracy_table,
    compute_cgcot_diagnostics,
    bootstrap_ci,
    mcnemar_test,
    cohens_d,
    load_results,
)
import numpy as np

PAPER_PATH = Path("paper/acl2026_srw.tex")
METHOD_LABELS = {"cot": "CoT", "self_consistency": "SC (k=3)", "pot": "PoT", "cgcot": "CG-CoT (Ours)"}
BENCH_ORDER = ["gsm8k", "math", "folio"]


def fill_main_table(tex: str, df: pd.DataFrame) -> str:
    """Replace the placeholder main results table with actual numbers."""
    acc = compute_accuracy_table(df)
    model = acc["model"].iloc[0]
    acc = acc[acc["model"] == model]

    # Build rows
    rows = []
    for method in ["cot", "self_consistency", "pot", "cgcot"]:
        label = METHOD_LABELS[method]
        vals = []
        for bench in BENCH_ORDER:
            row = acc[(acc["benchmark"] == bench) & (acc["method"] == method)]
            if len(row):
                vals.append(f"{row['accuracy_pct'].values[0]:.1f}")
            else:
                vals.append("--")
        avg = acc[acc["method"] == method]["accuracy_pct"].mean()
        vals.append(f"{avg:.1f}")
        rows.append((label, vals))

    # Find best per column
    best = []
    for col_idx in range(4):
        max_val = -1
        for _, vals in rows:
            try:
                v = float(vals[col_idx])
                if v > max_val:
                    max_val = v
            except ValueError:
                pass
        best.append(max_val)

    # Format with bold for best
    table_rows = []
    for label, vals in rows:
        formatted = []
        for i, v in enumerate(vals):
            try:
                if float(v) == best[i]:
                    formatted.append(f"\\textbf{{{v}}}")
                else:
                    formatted.append(v)
            except ValueError:
                formatted.append(v)
        prefix = "\\midrule\n" if label == "CG-CoT (Ours)" else ""
        table_rows.append(f"{prefix}{label} & {' & '.join(formatted)} \\\\")

    new_table = "\n".join(table_rows)

    # Replace placeholder rows
    old_pattern = (
        r"CoT & -- & -- & -- & -- \\\\\n"
        r"SC \(k=3\) & -- & -- & -- & -- \\\\\n"
        r"PoT & -- & -- & -- & -- \\\\\n"
        r"\\midrule\n"
        r"CG-CoT \(Ours\) & -- & -- & -- & -- \\\\"
    )
    tex = re.sub(old_pattern, new_table, tex)
    return tex


def fill_diagnostics_table(tex: str, df: pd.DataFrame) -> str:
    """Replace diagnostics placeholder."""
    diag = compute_cgcot_diagnostics(df)
    if diag.empty:
        return tex

    rows = []
    for bench in BENCH_ORDER:
        row = diag[diag["benchmark"] == bench]
        if len(row):
            r = row.iloc[0]
            rows.append(f"{bench.upper()} & {r['verification_rate']:.1f}\\% & {r['disagreement_rate']:.1f}\\% & {r['recovery_rate']:.1f}\\% \\\\")
        else:
            rows.append(f"{bench.upper()} & -- & -- & -- \\\\")

    new_rows = "\n".join(rows)

    old_pattern = (
        r"GSM8K & -- & -- & -- \\\\\n"
        r"MATH & -- & -- & -- \\\\\n"
        r"FOLIO & -- & -- & -- \\\\"
    )
    tex = re.sub(old_pattern, new_rows, tex)
    return tex


def fill_abstract(tex: str, df: pd.DataFrame) -> str:
    """Replace X%, Y%, Z% placeholders in abstract."""
    diag = compute_cgcot_diagnostics(df)
    acc = compute_accuracy_table(df)
    model = acc["model"].iloc[0]

    # Error detection rate (avg across benchmarks)
    if not diag.empty:
        edr = diag["disagreement_rate"].mean()
        tex = tex.replace("\\textbf{X\\%}", f"\\textbf{{{edr:.1f}\\%}}")
        recov = diag["recovery_rate"].mean()
        tex = tex.replace("\\textbf{Y\\%}", f"\\textbf{{{recov:.1f}\\%}}")

    # Best improvement over CoT
    acc_model = acc[acc["model"] == model]
    cot_accs = acc_model[acc_model["method"] == "cot"].set_index("benchmark")["accuracy_pct"]
    cgcot_accs = acc_model[acc_model["method"] == "cgcot"].set_index("benchmark")["accuracy_pct"]
    max_improvement = 0
    for bench in BENCH_ORDER:
        if bench in cot_accs.index and bench in cgcot_accs.index:
            imp = cgcot_accs[bench] - cot_accs[bench]
            max_improvement = max(max_improvement, imp)
    tex = tex.replace("\\textbf{Z percentage points}", f"\\textbf{{{max_improvement:.1f} percentage points}}")

    return tex


def main():
    df = load_results()
    tex = PAPER_PATH.read_text()

    tex = fill_abstract(tex, df)
    tex = fill_main_table(tex, df)
    tex = fill_diagnostics_table(tex, df)

    PAPER_PATH.write_text(tex)
    print(f"Paper updated at {PAPER_PATH}")


if __name__ == "__main__":
    main()
