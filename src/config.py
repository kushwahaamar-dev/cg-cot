"""Global configuration: paths, model names, constants."""

from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
PAPER_DIR = ROOT_DIR / "paper"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── Models (Ollama Only) ────────────────────────────────────────────
MODEL_REASONER = "qwen2.5:7b"         # Primary reasoning
MODEL_VERIFIER = "qwen2.5-coder:7b"     # Specialized Code Verifier
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Experiment settings ─────────────────────────────────────────────
TEMPERATURE = 0.0                       # deterministic for main runs
SC_TEMPERATURE = 0.7                    # self-consistency sampling
SC_K = 3                                # self-consistency samples
MAX_TOKENS = 2048
CODE_EXEC_TIMEOUT_S = 5                 # sandbox timeout
MAX_REGEN_ATTEMPTS = 1                  # CG-CoT regeneration budget

# ── Benchmark sample sizes (Per Run) ────────────────────────────────
# Use these for rapid iteration; final run should use matched sets
BENCHMARK_SIZES = {
    "gsm8k": 40,
    "math": 40,
    "folio": 40,
}

# ── Incremental save ────────────────────────────────────────────────
FLUSH_EVERY = 5                         # rows before CSV flush
SEED = 42                               # reproducibility
