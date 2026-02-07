#!/usr/bin/env python3
"""Main experiment orchestrator — run all benchmarks × models × methods.

Usage:
    # Full run with Ollama (Qwen Reasoner + Mistral Verifier)
    python run_experiment.py --model qwen2.5:7b

    # Smoke test
    python run_experiment.py --smoke

    # Single benchmark / method
    python run_experiment.py --benchmarks gsm8k --methods cot symmetric_cgcot
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

from tqdm import tqdm

from src.config import (
    FLUSH_EVERY,
    MODEL_REASONER,
    RESULTS_DIR,
    SEED,
)
from src.benchmarks import load_gsm8k, load_math, load_folio
from src.llm_client import make_client
from src.methods import (
    ChainOfThought,
    SymmetricCGCoT,
)
from src.models import ExperimentRow

# Configure logging
# Root logger captures DEBUG, but handlers filter
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# 1. Console: INFO only (clean progress)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S"
))
root_logger.addHandler(console_handler)

# 2. File: DEBUG (full traces)
file_handler = logging.FileHandler("experiment.log", mode="a")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
))
root_logger.addHandler(file_handler)

logger = logging.getLogger("experiment")

# ── Registry ────────────────────────────────────────────────────────

BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "folio": load_folio,
}

METHOD_CLASSES = {
    "cot": ChainOfThought,             # Baseline
    # "pot": ProgramOfThought,         # Optional baseline
    # "cgcot": CodeGroundedCoT,        # Legacy (Unidirectional)
    "symmetric_cgcot": SymmetricCGCoT, # Novelty (Bidirectional)
}


# ── CSV helpers ─────────────────────────────────────────────────────

def _csv_path(model_name: str, suffix: str = "") -> Path:
    safe_model = model_name.replace("/", "_").replace(":", "_")
    if suffix:
        return RESULTS_DIR / f"results_ollama_{safe_model}_{suffix}.csv"
    return RESULTS_DIR / f"results_ollama_{safe_model}.csv"


def _load_existing(path: Path) -> set[tuple[str, str, str]]:
    """Return set of (benchmark, problem_id, method) already completed."""
    done = set()
    if path.exists():
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["benchmark"], row["problem_id"], row["method"]))
    return done


def _init_csv(path: Path) -> None:
    """Write header if file doesn't exist."""
    if not path.exists():
        fields = list(ExperimentRow.model_fields.keys())
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def _flush_rows(path: Path, rows: list[ExperimentRow]) -> None:
    """Append rows to CSV."""
    if not rows:
        return
    fields = list(ExperimentRow.model_fields.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        for row in rows:
            writer.writerow(row.model_dump())


# ── Main loop ───────────────────────────────────────────────────────

def run(
    model_name: str,
    benchmarks: list[str],
    methods: list[str],
    smoke: bool = False,
    suffix: str = "",
    limit: int | None = None,
):
    logger.info("Starting experiment. SEED=%d Model=%s Suffix=%s Limit=%s", SEED, model_name, suffix, limit)
    
    # Primary client (Reasoning)
    llm = make_client("ollama", model=model_name)
    
    csv_path = _csv_path(model_name, suffix)
    _init_csv(csv_path)
    done = _load_existing(csv_path)
    # Also load from the main file to avoid re-doing work if we are splitting just now?
    # Actually, no. If we split, we want to write new results to new file.
    # Duplicates across files can be handled by analysis.py (it dedups).
    
    sample_size = 5 if smoke else None

    for bench_name in benchmarks:
        loader = BENCHMARK_LOADERS[bench_name]
        problems = loader(n=sample_size)
        
        if limit and limit < len(problems):
            problems = problems[:limit]
            
        logger.info("Benchmark %s: %d problems (capped by limit=%s)", bench_name, len(problems), limit)

        for method_name in methods:
            if method_name not in METHOD_CLASSES:
                logger.warning("Method %s not found, skipping", method_name)
                continue
                
            method_cls = METHOD_CLASSES[method_name]
            method = method_cls()
            skipped = 0
            buffer: list[ExperimentRow] = []

            desc = f"{bench_name}/{method_name}"
            for problem in tqdm(problems, desc=desc, unit="prob"):
                key = (problem.benchmark.value, problem.id, method.name.value)
                if key in done:
                    skipped += 1
                    continue

                row = method.solve(problem, llm)
                buffer.append(row)

                if len(buffer) >= FLUSH_EVERY:
                    _flush_rows(csv_path, buffer)
                    buffer.clear()

            # Flush remaining
            _flush_rows(csv_path, buffer)
            buffer.clear()

            logger.info(
                "%s/%s: completed (skipped %d). Tokens: %d",
                bench_name, method_name, skipped, llm.total_tokens_used,
            )

    logger.info("All done. Results → %s", csv_path)
    logger.info("Reasoning Usage: %s", llm.usage_summary)


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Symmetric CG-CoT experiment runner (Ollama)")
    parser.add_argument(
        "--model", default=MODEL_REASONER,
        help=f"Reasoning model name (default: {MODEL_REASONER})",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k", "math", "folio"],
        choices=["gsm8k", "math", "folio"],
        help="Which benchmarks to run",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["cot", "symmetric_cgcot"],
        help="Which methods to run (default: cot, symmetric_cgcot)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 5 problems per benchmark",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Suffix for output filename (enables parallel runs protection)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of problems per benchmark",
    )
    args = parser.parse_args()

    run(
        model_name=args.model,
        benchmarks=args.benchmarks,
        methods=args.methods,
        smoke=args.smoke,
        suffix=args.suffix,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
