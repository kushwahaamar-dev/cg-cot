#!/usr/bin/env python3
"""Generate all figures and statistical tables from experiment results.

Usage:
    python run_analysis.py
"""

import logging

from src.analysis import full_analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    full_analysis()
