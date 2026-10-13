#!/usr/bin/env python3
"""Materialise the stored execution record from the original notebook.

This script does not run GPT-2. It extracts embedded PNG figures and preserves
the checked, printed numerical results in machine-readable files so readers can
audit exactly what this checkout contains.
"""
from __future__ import annotations

import base64
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "01_pynb.ipynb"
RESULTS = ROOT / "results"

BASELINES = [
    {"dataset": "D0", "clean_logit_difference": 3.60, "corrupted_logit_difference": -0.32, "io_over_subject_accuracy": 1.00},
    {"dataset": "D1", "clean_logit_difference": 3.42, "corrupted_logit_difference": -0.19, "io_over_subject_accuracy": 1.00},
    {"dataset": "D2", "clean_logit_difference": 3.88, "corrupted_logit_difference": -0.23, "io_over_subject_accuracy": 0.99},
]
HEADS = [
    (8, 6, .379, .350, .409, .369, .368),
    (5, 5, .281, .259, .307, .089, .303),
    (8, 10, .279, .255, .306, .329, .290),
    (7, 9, .239, .224, .256, .220, .270),
    (9, 9, .218, .192, .247, .126, .170),
    (6, 9, .171, .145, .195, .122, .150),
    (3, 0, .148, .132, .169, .103, .150),
    (7, 3, .121, .107, .134, .117, .090),
    (10, 0, .101, .091, .113, .099, .140),
    (5, 9, .093, .066, .123, .238, .070),
]
EVIDENCE = {
    "provenance": "Transcribed from printed outputs in 01_pynb.ipynb; this script does not rerun the model.",
    "baselines": BASELINES,
    "ranking_agreement": {"d0_patch_vs_zero_spearman": .485, "d0_vs_d1_patch_spearman": .933, "d0_vs_d2_patch_spearman": .912, "d0_d1_top10_overlap": 10, "d0_d2_top10_overlap": 10},
    "cumulative_patching_fraction_gap_removed": [{"k": k, "top_k": top, "random_k": random} for k, top, random in [(1,.38,.00),(2,.52,.06),(3,.69,-.00),(5,.83,.10),(8,1.00,.09),(10,.97,.11),(15,1.00,.13),(20,1.02,.32)]],
    "mlp_patching_top5": [{"layer": x, "score": y} for x, y in [(0,.973),(5,.224),(4,.131),(2,.095),(1,.082)]],
    "mlp_zero_ablation_top5": [{"layer": x, "score": y} for x, y in [(0,1.253),(5,.321),(7,.246),(6,.214),(4,.209)]],
    "caveat": "Scores can exceed one under this normalized mean-gap metric. They are intervention effect sizes, not proportions of a disjoint additive circuit."
}

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    figures = RESULTS / "figures"
    figures.mkdir(exist_ok=True)
    EVIDENCE["top_d0_patch_heads"] = [
        {"layer": l, "head": h, "d0_patch_score": patch, "bootstrap_95_ci_low": low,
         "bootstrap_95_ci_high": high, "d0_zero_score": zero, "d1_patch_score": d1}
        for l, h, patch, low, high, zero, d1 in HEADS
    ]
    (RESULTS / "executed_notebook_evidence.json").write_text(json.dumps(EVIDENCE, indent=2) + "\n")
    with (RESULTS / "head_patching_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVIDENCE["top_d0_patch_heads"][0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(EVIDENCE["top_d0_patch_heads"])
    with (RESULTS / "baseline_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASELINES[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(BASELINES)
    notebook = json.loads(NOTEBOOK.read_text())
    image_count = 0
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                image_count += 1
                if isinstance(png, list): png = "".join(png)
                (figures / f"notebook_figure_{image_count:02d}.png").write_bytes(base64.b64decode(png))
    (RESULTS / "README.md").write_text(
        f"# Stored execution evidence\n\nGenerated from `{NOTEBOOK.name}` without rerunning GPT-2. "
        f"It contains {image_count} figures embedded in the original notebook, plus checked printed summaries. "
        "See `../REPORT.md` for interpretation and limitations.\n"
    )
    print(f"Wrote evidence tables and {image_count} embedded notebook figures to {RESULTS.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
