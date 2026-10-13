# Project log and evidence status

## Original notebook run (recorded in `01_pynb.ipynb`)

The original notebook records an executed GPT-2 Small experiment on a Tesla T4 using Python 3.13.15, PyTorch 2.11.0+cu128, TransformerLens 4.0.0, and Transformers 5.17.0. Its printed output records D0 (100 prompts, seed 0), D1 (held-out template, seed 1), and D2 (held-out lexical list, seed 2), head patching and ablation sweeps, bootstrapped head intervals, cumulative patching, attention visualisations, and whole-MLP interventions.

The authoritative numerical transcription is now `results/executed_notebook_evidence.json`; it was copied from printed output cells rather than from prose summaries.

## Documentation audit — 13 October 2026

The repository was audited without rerunning GPT-2. The audit found that `README.md`, `LOG.md`, and a notebook-generated prose summary made conflicting D2 claims: one summary said rank correlation 0.781 and 9/10 overlap, whereas the executed output reports 0.912 and 10/10. The latter is used because it appears in the notebook’s direct printed execution output. Earlier documentation also referred to `results/head_sweep.pt`, charts, and a zip file that are absent from the checkout.

Actions completed:

- Replaced unsupported “phase complete” language with an evidence-status statement.
- Created an extraction script and machine-readable CSV/JSON record from stored notebook outputs.
- Extracted the four non-empty embedded notebook figures into `results/figures/`.
- Wrote `RESULTS.md` and `REPORT.md`, explicitly separating stored measurements from future experiments.
- Preserved the original notebook untouched as the primary historical artifact.

## Reproducibility status

**Evidence-preserving, not fully reproducible from this checkout.** The code for the original run is present in the notebook, but raw activation caches, per-prompt outcomes, generated figures/tensors, dependency lockfile, and the original runtime are not. A future rerun must create a new dated evidence bundle rather than overwrite these stored results.
