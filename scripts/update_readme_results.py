"""
scripts/update_readme_results.py

Copy result tables from results/ into README.md between the RESULTS markers,
so the README can only show numbers that exist in a results file.

    python scripts/update_readme_results.py [--tests "N passed"]
"""
from __future__ import annotations

import argparse
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
HEADLINE = os.path.join("results", "rl", "grid_l125_b06")
SOURCES = {
    "rl_overview": os.path.join("results", "rl", "overview.md"),
    "training": os.path.join(HEADLINE, "summary.md"),
    "training_qos": os.path.join(HEADLINE, "qos.md"),
    "bounds": os.path.join("results", "bounds", "training.md"),
    "benchmark": os.path.join("results", "benchmark", "summary.md"),
    "benchmark_qos": os.path.join("results", "benchmark", "qos.md"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tests", default=None, help='e.g. "95 passed" (last pytest summary)')
    a = p.parse_args()

    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()

    blocks = {}
    for key, rel in SOURCES.items():
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            blocks[key] = f.read().strip()
    if a.tests:
        blocks["tests"] = f"Last run: **{a.tests}**."

    for key, body in blocks.items():
        pattern = re.compile(rf"(<!-- RESULTS:{key}:start -->\n).*?(<!-- RESULTS:{key}:end -->)", re.S)
        if not pattern.search(text):
            raise SystemExit(f"README.md has no RESULTS:{key} markers")
        text = pattern.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)

    with open(readme_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("README.md updated:", ", ".join(blocks))


if __name__ == "__main__":
    main()
