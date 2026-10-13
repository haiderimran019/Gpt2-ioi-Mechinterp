# GPT-2 Small IOI: an audited intervention study

This repository records a first mechanistic-interpretability study of GPT-2 Small on **Indirect Object Identification (IOI)**. The original work is in [`01_pynb.ipynb`](01_pynb.ipynb). This revision makes a crucial distinction: the notebook contains evidence of a completed run, but this checkout does **not** contain its raw tensors or a runnable environment. The reported values below are therefore **stored notebook outputs**, not a newly rerun experiment.

The full paper is [`REPORT.md`](REPORT.md); the concise evidence audit is [`RESULTS.md`](RESULTS.md).

## What the stored run supports

On 100 D0 prompts, the notebook reported a clean IOI logit difference of 3.60, a corrupted value of -0.32, and 100% IO-over-subject accuracy. Corrupted-activation patching ranked L8H6 highest (0.379 fraction of the clean-corrupted gap removed; bootstrap 95% CI [0.350, 0.409]). The ten highest-ranked heads were reproduced exactly in the notebook’s held-out template and lexical-set sweeps (rank correlations 0.933 and 0.912 respectively).

It also reports broad MLP-output patching effects, especially at layer 0 (0.973) and layer 5 (0.224). This is evidence that those **whole MLP outputs are causally consequential under the stated intervention**. It is not evidence that a particular MLP mediates a particular head: no head-by-MLP factorial or path-patching experiment was run.

## What this repository does not claim

- No independently repeated execution was performed in this checkout.
- No new IOI circuit, novelty, or head–MLP mechanism is claimed.
- The results are not sufficient to identify a causal edge from a named attention head to an MLP.
- The missing raw outputs prevent recomputation of CIs, p-values, rank uncertainty, or alternative statistics.

## Files

```text
01_pynb.ipynb                         original executed notebook and embedded figures
configs/executed_notebook_manifest.json exact study record and evidence limits
scripts/extract_notebook_evidence.py   extracts the stored evidence; does not run GPT-2
results/                               generated CSV/JSON evidence tables and notebook figures
RESULTS.md                             reconciled results and error audit
REPORT.md                              full research-paper draft
LOG.md                                 chronological project and audit record
literature/                            reading notes and literature index
```

## Quick audit

```bash
python3 scripts/extract_notebook_evidence.py
```

This only materialises existing notebook outputs. It does not download a model, alter the original notebook, or create fresh measurements.

## Re-running later

`requirements.txt` records the package family used by the original notebook. Before treating a rerun as a replication, pin an environment that can actually install the historical CUDA/Tesla-T4 stack, save per-prompt values and activations, and run the registered factorial head-by-MLP design in the paper’s Future Work section.
