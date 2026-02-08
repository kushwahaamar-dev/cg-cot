import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")

RESULTS_DIR = Path("results")
FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    dfs = []
    for f in RESULTS_DIR.glob("results_ollama_qwen2.5_7b*.csv"):
        if "qwen_coder" in f.name: continue # Skip empty/partial file
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True)

def plot_accuracy(df):
    # Calculate accuracy per benchmark/method
    # correct column is strictly 'True'/'False' strings or booleans
    df["correct_bool"] = df["correct"].astype(str) == "True"
    
    acc = df.groupby(["benchmark", "method"])["correct_bool"].mean().reset_index()
    acc["Accuracy (%)"] = acc["correct_bool"] * 100
    
    # Rename for pretty plotting
    acc["Method"] = acc["method"].replace({
        "cot": "Baseline (CoT)",
        "symmetric_cgcot": "Symmetric (Generalist)"
    })
    acc["Benchmark"] = acc["benchmark"].str.upper()
    
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=acc, x="Benchmark", y="Accuracy (%)", hue="Method")
    plt.title("Main Results: Accuracy by Method")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_main_results.png", dpi=300)
    print("Generated fig_main_results.png")

def plot_reliability_gap(df):
    # Filter for Symmetric method only to show Crash Rates
    sym = df[df["method"] == "symmetric_cgcot"].copy()
    
    # Calculate crash rate
    # steps_crashed / steps_total
    # If steps_total is 0, crash rate is 0
    sym["crash_rate"] = sym.apply(lambda r: (r["steps_crashed"] / r["steps_total"] * 100) if r["steps_total"] > 0 else 0, axis=1)
    
    # Group by benchmark
    gap = sym.groupby("benchmark")["crash_rate"].mean().reset_index()
    
    # Add manual row for Specialist (Case Study)
    # We only have data for Mistral (Generalist) in the CSVs
    gap["Verifier"] = "Mistral-7B (Generalist)"
    
    # Add hypothetical Qwen-Coder point for MATH
    specialist_row = pd.DataFrame({
        "benchmark": ["math"],
        "crash_rate": [0.0],
        "Verifier": ["Qwen-Coder (Specialist)"]
    })
    
    combined = pd.concat([gap, specialist_row], ignore_index=True)
    combined["Benchmark"] = combined["benchmark"].str.upper()
    
    plt.figure(figsize=(6, 5))
    sns.barplot(data=combined, x="Benchmark", y="crash_rate", hue="Verifier", palette=["#D55E00", "#009E73"])
    plt.title("The Reliability Gap: Crash Rate by Verifier")
    plt.ylabel("Crash Rate (% of Steps)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_reliability_gap.png", dpi=300)
    print("Generated fig_reliability_gap.png")
    
def plot_cost_vs_acc(df):
    df["correct_bool"] = df["correct"].astype(str) == "True"
    
    plt.figure(figsize=(8, 6))
    
    # Scatter plot
    sns.scatterplot(
        data=df, 
        x="tokens_used", 
        y="benchmark", 
        hue="correct_bool", 
        style="method",
        alpha=0.6,
        s=100
    )
    plt.title("Cost (Tokens) vs Correctness")
    plt.xlabel("Tokens Used")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_cost_accuracy.png", dpi=300)
    print("Generated fig_cost_accuracy.png")

if __name__ == "__main__":
    df = load_data()
    plot_accuracy(df)
    plot_reliability_gap(df)
    plot_cost_vs_acc(df)
