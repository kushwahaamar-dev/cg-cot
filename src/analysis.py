"""Statistical analysis and figure generation for the CG-CoT paper."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .config import FIGURES_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.1)

METHOD_ORDER = ["cot", "symmetric_cgcot"]
METHOD_LABELS = {
    "cot": "CoT",
    "symmetric_cgcot": "Symmetric CG-CoT",
}
BENCH_LABELS = {"gsm8k": "GSM8K", "math": "MATH", "folio": "FOLIO"}
PALETTE = {
    "CoT": "#5B9BD5",
    "Symmetric CG-CoT": "#E74C3C",
}


# ── Data loading ────────────────────────────────────────────────────

def load_results(matched: bool = True) -> pd.DataFrame:
    """Concatenate all results CSVs into one DataFrame."""
    csvs = list(RESULTS_DIR.glob("results_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No results CSVs found in {RESULTS_DIR}")
    dfs = [pd.read_csv(p) for p in csvs]
    df = pd.concat(dfs, ignore_index=True)

    # De-duplicate: keep the last row for each (benchmark, problem_id, method, model)
    df = df.drop_duplicates(subset=["benchmark", "problem_id", "method", "model"], keep="last")

    if matched:
        # Keep only problems where all methods ran
        all_methods = set(METHOD_ORDER)
        keep_ids = []
        for (bench, model), grp in df.groupby(["benchmark", "model"]):
            for pid, pgrp in grp.groupby("problem_id"):
                if set(pgrp["method"].values) >= all_methods:
                    keep_ids.append((bench, pid, model))
        
        # If no matched data (e.g. partial run), allow standard load with warning
        if not keep_ids:
            logger.warning("No fully matched intersection found. Returning all data.")
        else:
            keys = pd.DataFrame(keep_ids, columns=["benchmark", "problem_id", "model"])
            df = df.merge(keys, on=["benchmark", "problem_id", "model"])

    df["method_label"] = df["method"].map(METHOD_LABELS)
    df["bench_label"] = df["benchmark"].map(BENCH_LABELS)
    return df


# ── Metrics ─────────────────────────────────────────────────────────

def compute_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per (benchmark, method, model)."""
    grouped = (
        df.groupby(["benchmark", "method", "model"])["correct"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    grouped.rename(columns={"mean": "accuracy", "count": "n", "sum": "n_correct"}, inplace=True)
    grouped["accuracy_pct"] = (grouped["accuracy"] * 100).round(1)
    return grouped


def compute_cgcot_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """CG-CoT–specific diagnostics."""
    cg = df[df["method"] == "symmetric_cgcot"].copy()
    if cg.empty:
        return cg
    
    # Ensure columns exist (backwards compatibility)
    if "steps_crashed" not in cg.columns:
        cg["steps_crashed"] = 0

    metrics = cg.groupby(["benchmark", "model"]).agg(
        steps_total=("steps_total", "sum"),
        steps_verified=("steps_verified", "sum"),
        steps_disagreed=("steps_disagreed", "sum"),
        steps_recovered=("steps_recovered", "sum"),
        steps_crashed=("steps_crashed", "sum"),
        n=("correct", "count"),
    ).reset_index()
    
    metrics["crash_rate"] = (metrics["steps_crashed"] / metrics["steps_total"].clip(lower=1) * 100).round(1)
    metrics["verification_rate"] = (metrics["steps_verified"] / metrics["steps_total"].clip(lower=1) * 100).round(1)
    metrics["disagreement_rate"] = (metrics["steps_disagreed"] / metrics["steps_verified"].clip(lower=1) * 100).round(1)
    metrics["recovery_rate"] = (metrics["steps_recovered"] / metrics["steps_disagreed"].clip(lower=1) * 100).round(1)
    
    # Add Crash Rate
    if "steps_crashed" in df.columns:
         metrics["steps_crashed"] = df["steps_crashed"].sum()
         metrics["crash_rate"] = (metrics["steps_crashed"] / metrics["steps_total"].clip(lower=1) * 100).round(1)
    
    return metrics


def generate_latex_table(df: pd.DataFrame):
    """Generate LaTeX table rows for the results."""
    # Filter for GSM8K for now as it's the primary focus
    df_gsm = df[df["benchmark"] == "gsm8k"]
    if df_gsm.empty:
        return
    
    def get_metrics(sub_df):
        if sub_df.empty: return {}
        m = {}
        m["accuracy_pct"] = (sub_df["correct"].mean() * 100).round(1)
        m["avg_tokens"] = sub_df["tokens_used"].mean()
        m["avg_latency"] = sub_df["latency_s"].mean()
        
        # Method specific
        if "steps_recovered" in sub_df.columns:
            recovered = sub_df["steps_recovered"].sum()
            disagreed = sub_df["steps_disagreed"].sum()
            m["recovery_rate"] = (recovered / disagreed * 100).round(1) if disagreed > 0 else 0.0
            
        if "steps_crashed" in sub_df.columns:
            crashed = sub_df["steps_crashed"].sum()
            total = sub_df["steps_total"].sum()
            m["crash_rate"] = (crashed / total * 100).round(1) if total > 0 else 0.0
        return m

    # COT
    cot_metrics = get_metrics(df_gsm[df_gsm["method"] == "cot"])
    
    # SYM
    sym_metrics = get_metrics(df_gsm[df_gsm["method"] == "symmetric_cgcot"])
    
    def fmt(val, unit=""):
        try:
            return f"{float(val):.1f}{unit}"
        except (ValueError, TypeError):
            return "N/A"
    
    print("\n=== LATEX TABLE ROWS ===")
    print(f"Accuracy & {fmt(cot_metrics.get('accuracy_pct'), '\\%')} & {fmt(sym_metrics.get('accuracy_pct'), '\\%')} \\\\")
    print(f"Avg Tokens & {int(cot_metrics.get('avg_tokens', 0))} & {int(sym_metrics.get('avg_tokens', 0))} \\\\")
    print(f"Avg Latency & {fmt(cot_metrics.get('avg_latency', 0))}s & {fmt(sym_metrics.get('avg_latency', 0))}s \\\\")
    print(f"\\midrule")
    print(f"\\textit{{Recovery Rate}} & N/A & {fmt(sym_metrics.get('recovery_rate'), '\\%')} ({int(df_gsm[df_gsm['method']=='symmetric_cgcot']['steps_recovered'].sum())}/{int(df_gsm[df_gsm['method']=='symmetric_cgcot']['steps_disagreed'].sum())}) \\\\")
    print(f"\\textit{{Crash Rate}} & 0.0\\% & {fmt(sym_metrics.get('crash_rate'), '\\%')} \\\\")
    print("========================\n")

def report_system_errors(df: pd.DataFrame):
    """Report any system errors logged in the 'error' column."""
    if "error" not in df.columns:
        return
    
    errors = df[df["error"].notna() & (df["error"] != "")]
    if not errors.empty:
        print(f"\n=== SYSTEM ERRORS ({len(errors)}) ===")
        print(errors["error"].value_counts().to_string())
    else:
        print("\n=== SYSTEM ERRORS: None ===")



# ── Statistical tests ───────────────────────────────────────────────

def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    """Paired bootstrap for difference in means (a - b)."""
    rng = np.random.default_rng(42)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = (np.abs(diffs) < np.abs(diffs.mean())).mean()  # approximate two-sided p
    return float(diffs.mean()), float(lo), float(hi), float(p)


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray):
    """McNemar's test."""
    b01 = int(((a_correct == 1) & (b_correct == 0)).sum())
    b10 = int(((a_correct == 0) & (b_correct == 1)).sum())
    if b01 + b10 == 0:
        return 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return float(chi2), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size (Cohen's d)."""
    diff = a - b
    std = diff.std(ddof=1)
    return float(diff.mean() / (std + 1e-12))


# ── Figures ─────────────────────────────────────────────────────────

def fig_main_results(df: pd.DataFrame, save: bool = True):
    """Figure: Accuracy bar chart."""
    tbl = compute_accuracy_table(df)
    if tbl.empty: return None
    
    model = tbl["model"].iloc[0]
    tbl = tbl[tbl["model"] == model]

    fig, ax = plt.subplots(figsize=(9, 5))
    benchmarks = [b for b in ["gsm8k", "math", "folio"] if b in tbl["benchmark"].values]
    x = np.arange(len(benchmarks))
    width = 0.35
    methods = [m for m in METHOD_ORDER if m in tbl["method"].values]

    for j, method in enumerate(methods):
        label = METHOD_LABELS.get(method, method)
        accs = []
        for bench in benchmarks:
            row = tbl[(tbl["benchmark"] == bench) & (tbl["method"] == method)]
            accs.append(row["accuracy_pct"].values[0] if len(row) else 0)
        
        bars = ax.bar(x + j * width - width/2, accs, width, label=label, color=PALETTE.get(label, "#999"))
        for bar in bars:
             ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABELS.get(b, b) for b in benchmarks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right")
    ax.set_title(f"Symmetric Verification Results — {model}")
    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "fig_main_results.png"
        fig.savefig(path, dpi=200)
        logger.info("Saved %s", path)
    return fig


def fig_reliability_gap(df: pd.DataFrame, save: bool = True):
    """Figure: Reliability Gap (Crash Rate vs Recovery Rate)."""
    diag = compute_cgcot_diagnostics(df)
    if diag.empty: return None

    # Filter benchmarks
    benchmarks = [b for b in ["gsm8k", "math", "folio"] if b in diag["benchmark"].values]
    diag = diag.set_index("benchmark").reindex(benchmarks).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(benchmarks))
    width = 0.35

    # Data
    crash_rate = diag["crash_rate"].values
    recovery_rate = diag["recovery_rate"].values

    # Plot
    rects1 = ax.bar(x - width/2, crash_rate, width, label='Crash Rate (Reliability Gap)', color='#E74C3C', alpha=0.8)
    rects2 = ax.bar(x + width/2, recovery_rate, width, label='Recovery Rate (Success)', color='#2ECC71', alpha=0.8)

    # Labels
    ax.set_ylabel('Percentage (%)')
    ax.set_title('The Reliability Gap: Crash vs. Recovery')
    ax.set_xticks(x)
    ax.set_xticklabels([BENCH_LABELS.get(b, b) for b in benchmarks])
    ax.set_ylim(0, 100)
    ax.legend()
    
    # Add values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    if save:
        path = FIGURES_DIR / "fig_reliability_gap.png"
        fig.savefig(path, dpi=300)
        logger.info("Saved %s", path)
    return fig

def full_analysis():
    """Run full analysis pipeline."""
    df = load_results(matched=False)
    print(f"Loaded {len(df)} rows.")

    # 1. Main Results Table (Stdout + LaTeX)
    print("\n=== ACCURACY TABLE ===")
    acc_tbl = compute_accuracy_table(df)
    print(acc_tbl.to_string())
    
    # 2. System Errors
    report_system_errors(df)

    # 3. LaTeX Generation
    generate_latex_table(df)

    # 4. Figures
    print("\n=== SYMMETRIC DIAGNOSTICS ===")
    diag = compute_cgcot_diagnostics(df)
    print(diag.to_string())
    
    fig_main_results(df)
    fig_reliability_gap(df)
    
    print(f"\nFigures saved to {FIGURES_DIR}/")
    print("Analysis complete.")


# ── Report ──────────────────────────────────────────────────────────

def full_analysis(df: pd.DataFrame | None = None):
    """Run all analyses."""
    if df is None:
        df = load_results(matched=False) # Allow partials for now

    acc = compute_accuracy_table(df)
    print("\n=== ACCURACY TABLE ===")
    print(acc.to_string())

    print("\n=== SYSTEM ERRORS ===")
    report_system_errors(df)
    
    generate_latex_table(df)

    diag = compute_cgcot_diagnostics(df)
    if not diag.empty:
        print("\n=== SYMMETRIC DIAGNOSTICS ===")
        print(diag.to_string(index=False))

    fig_main_results(df)
    fig_reliability_gap(df)
    
    print(f"\nFigures saved to {FIGURES_DIR}/")
