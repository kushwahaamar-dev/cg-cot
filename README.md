# Symmetric Code-Grounded CoT

**Bridging the Reliability Gap in Open-Source LLCs via Bidirectional Verification**

> Target: ACL 2026 Student Research Workshop — Paper Deadline March 18, 2026

## Overview

Symmetric CG-CoT transforms verification from a "Code-as-Oracle" model to a **Peer Verification** model. Instead of blindly trusting code execution, we employ a dual-agent system where natural language logic can debug code, and corrected code can verify logic.

## Requirements

1. **Python 3.10+** (see `requirements.txt`)
2. **Ollama**: Local LLM runner.
   - Install from [ollama.com](https://ollama.com)
   - Pull the required models:
     ```bash
     ollama pull qwen2.5:7b    # Reasoning Agent
     ollama pull mistral:7b    # Verification/Coding Agent
     ```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run "Smoke Test" (5 problems, ensure pipeline works)
python run_experiment.py --smoke

# Run full experiment on GSM8K
python run_experiment.py --benchmarks gsm8k --methods cot symmetric_cgcot

# Generate figures + stats
python run_analysis.py
```

## Methodology

We compare:
1. **Chain-of-Thought (CoT)**: Standard baseline.
2. **Symmetric CG-CoT**:
   - **Step 1:** Reasoner generates CoT + Python Code.
   - **Step 2:** Code is executed in a sandbox.
   - **Step 3 (Consensus Check):** If Code $\neq$ Logic:
     - **Branch A (Debug Code):** Verifier checks code for bugs using Logic as a hint.
     - **Branch B (Regenerate Logic):** If code persists, Reasoner re-thinks the step.

## Benchmarks

| Benchmark | Domain | Metric |
|-----------|--------|--------|
| GSM8K | Arithmetic Word Problems | Exact Match |
| MATH | Competition Mathematics | Symbolic Equivalence (SymPy) |
| FOLIO | First-Order Logic | Label Match |
