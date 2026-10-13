# Results: audited notebook evidence

## Evidence source

Every number in this document comes from a printed output cell in [`01_pynb.ipynb`](01_pynb.ipynb). The notebook evidently ran on a Tesla T4, but neither raw tensors nor the referenced `results/` artifacts were committed. Accordingly, this is a **reconciled record of an existing execution**, not a rerun or independent replication.

Run `python3 scripts/extract_notebook_evidence.py` to create the accompanying CSV/JSON tables and recover the four embedded figures. The script never executes the model.

## Primary behavioral check

| Dataset | Role | Mean clean logit difference | Mean corrupted logit difference | IO > subject accuracy |
|---|---|---:|---:|---:|
| D0 | primary; 100 prompts | 3.60 | -0.32 | 100% |
| D1 | held-out template; 100 prompts | 3.42 | -0.19 | 100% |
| D2 | held-out lexical list; 100 prompts | 3.88 | -0.23 | 99% |

The metric is correct-indirect-object logit minus subject-name logit. “Corrupted” means the notebook’s counterfactual prompt construction, not random noise.

## Head interventions

For D0, corrupted activation patching and zero ablation agree moderately across 144 heads (Spearman \(\rho=0.485\)). The top ten patching heads have bootstrap 95% intervals recorded in the notebook:

| Head | Patch score | 95% bootstrap CI | Zero-ablation score | D1 patch score |
|---|---:|---:|---:|---:|
| L8H6 | 0.379 | [0.350, 0.409] | 0.369 | 0.368 |
| L5H5 | 0.281 | [0.259, 0.307] | 0.089 | 0.303 |
| L8H10 | 0.279 | [0.255, 0.306] | 0.329 | 0.290 |
| L7H9 | 0.239 | [0.224, 0.256] | 0.220 | 0.270 |
| L9H9 | 0.218 | [0.192, 0.247] | 0.126 | 0.170 |
| L6H9 | 0.171 | [0.145, 0.195] | 0.122 | 0.150 |
| L3H0 | 0.148 | [0.132, 0.169] | 0.103 | 0.150 |
| L7H3 | 0.121 | [0.107, 0.134] | 0.117 | 0.090 |
| L10H0 | 0.101 | [0.091, 0.113] | 0.099 | 0.140 |
| L5H9 | 0.093 | [0.066, 0.123] | 0.238 | 0.070 |

The patching rank was robust within this narrow test: D0–D1 \(\rho=0.933\) with 10/10 top-head overlap; D0–D2 \(\rho=0.912\) with 10/10 overlap. These are descriptive robustness checks, not a generalization guarantee beyond the two templates and curated lexical lists.

Cumulative patching also favored the top-ranked heads over a notebook random baseline: at \(k=1,5,8\), reported fractions of the mean gap removed were 0.38, 0.83, and 1.00, compared with 0.00, 0.10, and 0.09 for the random sets. Values slightly above one at larger \(k\) reflect the normalized mean-gap metric and non-additive interactions; they should not be read as a partition of the circuit.

## MLP interventions

The notebook’s whole-output MLP patching sweep ranked L0 (0.973), L5 (0.224), L4 (0.131), L2 (0.095), and L1 (0.082). Its zero-ablation sweep ranked L0 (1.253), L5 (0.321), L7 (0.246), L6 (0.214), and L4 (0.209).

This establishes a useful **follow-up target**, not an interaction discovery. Replacing/zeroing a whole MLP output tests the MLP’s total effect and can cause distribution shift. Because the stored study never patches a head and MLP jointly or tests a route between them, it cannot tell whether L8H6, for example, works through L5, works independently, or is compensated elsewhere.

## Corrections to old prose

- Use D2 \(\rho=0.912\), 10/10 overlap—not the conflicting 0.781, 9/10 claim found in a notebook prose cell.
- Do not claim external result files exist: the earlier documentation named files absent from this checkout.
- Do not describe the work as a newly verified complete circuit or an original head–MLP finding.
