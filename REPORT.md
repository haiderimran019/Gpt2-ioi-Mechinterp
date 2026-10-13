# Robustness of component-level interventions in GPT-2 Small IOI

## An audited report based on an existing notebook execution

### Abstract

Mechanistic interpretability asks which internal computations produce a model behavior. Indirect Object Identification (IOI) in GPT-2 Small is a useful testbed because prior work identified a multi-head circuit using causal interventions. This report audits and interprets an existing notebook study of IOI rather than claiming a new model run. The stored output records 100 primary prompts (D0), a held-out wording template (D1), and a held-out lexical set (D2). On D0, GPT-2 Small had a mean correct-indirect-object minus subject logit difference of 3.60 and 100% pairwise accuracy. Corrupted-activation patching ranked L8H6 highest (0.379 of the clean-to-corrupted gap removed; bootstrap 95% CI [0.350, 0.409]). The patching ranking was stable across D1 (Spearman 0.933; 10/10 top-head overlap) and D2 (0.912; 10/10 overlap). Whole-MLP interventions found particularly large effects at MLP0 and smaller effects at MLP5. These results reproduce a familiar localized-intervention pattern within a narrow synthetic distribution. They do **not** identify a new circuit or show that an MLP causally mediates an attention head: the notebook did not run factorial head-by-MLP or path-patching tests, and raw per-prompt outputs are missing. The main contribution of this revision is an honest, machine-readable evidence record and a precise next experiment.

## 1. Introduction

Large language models can perform simple relational tasks without being explicitly programmed with their rules. The interesting question is not only whether the answer is correct, but what computation inside the network produces it. In the IOI task, a prompt such as “When Mary and John went to the store, John gave a drink to …” should be completed with *Mary*, not the final-clause subject *John*. Wang et al. studied this behavior in GPT-2 Small and proposed a 26-head, seven-class circuit supported by interventions [1].

This repository began as a learning replication of that setting. Its original notebook runs causal interventions on attention heads and MLP outputs, then tests head-ranking stability after modest wording and vocabulary changes. The motivating question for the current report is deliberately narrower than “discover the IOI circuit”:

> Within the notebook’s controlled IOI distribution, are head-level corrupted-patching rankings stable under held-out templates and lexical lists, and do existing whole-MLP interventions justify a specific head–MLP interaction claim?

The answer is two-part. The stored results support the narrow ranking-stability observation. They do not support a head–MLP mechanism claim. Making this distinction matters: an MLP may be important under a whole-layer intervention without mediating the effect of any particular head.

## 2. Prior work and research gap

Wang et al. [1] showed that IOI can be investigated with causal interventions rather than attention visualisation alone. They organized 26 heads into functional groups and evaluated faithfulness, completeness, and minimality. This establishes that a simplistic claim such as “we found important IOI heads” is neither novel nor a full circuit explanation.

Conmy et al. [2] formalized the difficulty of moving from important components to important *edges* with ACDC. Their central lesson here is that a component list is not a causal graph. The broader induction-head literature also cautions that ablations in networks containing MLPs can be non-additive: knocking out a head can shift an MLP’s input distribution or trigger compensation, so marginal head effects need not recover the mechanism [3].

These papers make an attainable gap for this small project: document whether intervention rankings generalize across simple distribution changes, and state exactly what remains necessary before arguing for head-to-MLP mediation. The present data can address the first part only. It is a replication/robustness contribution, not a novelty claim.

## 3. Evidence status and materials

The source artifact is [`01_pynb.ipynb`](01_pynb.ipynb). Its stored execution outputs identify Python 3.13.15, PyTorch 2.11.0+cu128, TransformerLens 4.0.0, Transformers 5.17.0, and a Tesla T4. The notebook includes code and four non-empty embedded figures. This checkout did not include the raw activation caches, per-prompt effects, generated result tensors, or the external `results/` files cited by its old prose.

Consequently, the results in this report are transcribed only from printed notebook outputs. [`scripts/extract_notebook_evidence.py`](scripts/extract_notebook_evidence.py) materialises that transcription as JSON and CSV and extracts the embedded PNGs; it never runs GPT-2. [`configs/executed_notebook_manifest.json`](configs/executed_notebook_manifest.json) records the exact scope and limitations. This is evidence preservation, not independent reproduction.

## 4. Methods of the original executed study

### 4.1 Task and datasets

Each prompt contains two candidate names. The dependent variable is the final-position logit for the correct indirect-object name minus the logit for the competing subject name. Positive values favor the IO answer.

- **D0**: 100 examples, seed 0, “When {first} and {second} went to the {place}, {subject} gave a {object} to”.
- **D1**: 100 examples, seed 1, held-out wording: “After {first} and {second} arrived at the {place}, {subject} handed a {object} to”.
- **D2**: 100 examples, seed 2, D0 syntax with the notebook’s disjoint list of curated names, places, and objects.

The task is a deliberately simple, templated contrast and should not be interpreted as broad pronoun-resolution ability.

### 4.2 Interventions

For each of 144 attention heads, the notebook ran two interventions: (a) replace the clean head activation with its activation on a corrupted prompt and (b) replace that head output with zero. It reports a normalized score: the fraction of the mean clean-to-corrupted logit gap removed by the intervention. It also evaluated cumulative patching of the top-ranked heads against a random-head baseline.

For MLPs, the notebook used the TransformerLens `blocks.{L}.mlp.hook_post` activation and performed the same two whole-output interventions for each of 12 layers. It bootstrapped D0 head patching scores to form 95% intervals, but the raw prompt-level scores needed to recalculate those intervals are not present.

### 4.3 Planned interaction test, not executed

The correct test of a head–MLP interaction is a two-factor design. For head intervention \(H\) and MLP intervention \(M\), calculate the effect on every prompt for none, \(H\) only, \(M\) only, and \(H+M\). The interaction is

\[
I = [y(H+M)-y(\varnothing)] - [y(H)-y(\varnothing)] - [y(M)-y(\varnothing)].
\]

Paired bootstrap intervals for \(I\), a multiple-testing correction across pre-registered pairs, seed replication, and held-out templates would test whether a combined perturbation differs from additive effects. Path patching would then be required to narrow a likely route. No result for \(I\) is reported here because this experiment was not run.

## 5. Results

### 5.1 Behavioral sanity check

| Split | Clean logit difference | Corrupted logit difference | Pairwise IO accuracy |
|---|---:|---:|---:|
| D0 | 3.60 | -0.32 | 100% |
| D1 | 3.42 | -0.19 | 100% |
| D2 | 3.88 | -0.23 | 99% |

The clean-corrupted contrast is large and has the expected direction in all three notebook splits. It shows that the generated prompts elicited the target behavior under the pairwise metric.

### 5.2 Head patching is concentrated and stable on these splits

The highest D0 corrupted-patching effect was L8H6, 0.379 (95% CI [0.350, 0.409]), followed by L5H5 (0.281 [0.259, 0.307]), L8H10 (0.279 [0.255, 0.306]), L7H9 (0.239 [0.224, 0.256]), and L9H9 (0.218 [0.192, 0.247]). The complete top-ten table is in [`RESULTS.md`](RESULTS.md) and [`results/head_patching_summary.csv`](results/head_patching_summary.csv) after extraction.

Patching versus zero-ablation rankings had only moderate agreement (Spearman 0.485), a useful warning that intervention choice matters. Six heads overlapped in their two top-ten lists: L8H6, L8H10, L7H9, L9H9, L6H9, and L5H9. In contrast, the *corrupted-patching* ranking was highly stable across the two narrow held-out splits: D0–D1 Spearman 0.933 and D0–D2 Spearman 0.912, with a complete top-ten overlap in each case.

Cumulative patching gives a complementary concentration check. The notebook reports top-k versus random-k fractions of the gap removed of 0.38 versus 0.00 at k=1, 0.83 versus 0.10 at k=5, and 1.00 versus 0.09 at k=8. This is consistent with a small set of selected heads being influential under this intervention. It does not prove their effects are additive; indeed, values near or above one at larger k show why summing marginal scores into a “percent of the circuit” would be misleading.

### 5.3 MLP effects motivate, but do not demonstrate, interaction

Whole-MLP patching was dominated by MLP0 (0.973), with MLP5 (0.224) and MLP4 (0.131) next. Whole-MLP zero ablation similarly ranked MLP0 (1.253), MLP5 (0.321), MLP7 (0.246), MLP6 (0.214), and MLP4 (0.209). The layer-0 result is especially sensitive to interpretation because replacing an early whole-layer output changes the input distribution for every downstream computation.

The available result is therefore: **these layer outputs were consequential under the original interventions.** It is not: “L8H6 works through MLP5,” “MLP0 combines named heads,” or “the project discovered an attention–MLP interaction.” The required joint intervention and path-patching evidence is absent.

## 6. Discussion

For a beginner-friendly reading, “patching” asks a counterfactual question: if we replace one component’s clean internal signal with the signal from a prompt where the roles have been swapped, how much does the answer become less correct? L8H6 passed this test especially strongly, and the same ordered set of heads remained prominent after small changes to wording and vocabulary. That makes the original analysis a useful replication-style demonstration that an importance ranking need not be an artifact of one exact sentence template.

The main scientific lesson is also a methodological one. Attention pattern pictures can help formulate hypotheses, but intervention effects are more meaningful than a dark square on a heatmap. Even interventions require care: a large marginal effect need not identify the full route or the only mechanism. The difference between the patching and zero-ablation rankings is direct evidence for this caution.

The plausible research direction is not an unsupported claim of novelty, but a registered interaction study focused on a very small preselected set: L8H6, L5H5, L8H10, L7H9, L9H9 crossed with MLP0, MLP4, and MLP5. A positive interaction robust to three or more fresh templates and multiple lexical draws would motivate path patching; a null result would be equally informative by showing that broad MLP importance does not imply mediation.

## 7. Limitations

1. The report is based on stored notebook output. It is not an independent rerun.
2. The sample is 100 prompts per split, drawn from templates and curated single-token lexical lists. No claim extends to natural text or other models.
3. Missing per-prompt values preclude recomputation of the bootstrap, formal tests between splits, seed variation, and inspection for outliers.
4. The clean/corrupted design and normalized mean-gap score are useful diagnostic choices but not unique causal estimands. Layer norm, redundancy, and compensation can distort marginal effects.
5. The study did not test pairwise head–MLP interactions, edges, neurons/features, or circuit faithfulness/completeness. It cannot establish an attention–MLP mechanism.
6. Previous documentation contained inconsistent D2 prose and referred to absent output files. This report resolves that conflict in favor of the direct execution output but cannot recover the missing raw artifacts.

## 8. Conclusion

The existing notebook provides credible stored evidence that a small group of GPT-2 Small heads has robust corrupted-patching rankings across two limited IOI distribution shifts, and that several whole MLP outputs matter under analogous interventions. That is a solid learning and replication result. It is not a new circuit discovery and does not yet answer the more ambitious question of how attention heads and MLPs interact. The repository is now organized so readers can inspect the original evidence, reproduce the documentation record, and see exactly what a future experiment must add.

## References

[1] K. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt. *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*. 2022. [arXiv:2211.00593](https://arxiv.org/abs/2211.00593).

[2] A. Conmy, A. Mavor-Parker, A. Lynch, S. Heimersheim, and A. Garriga-Alonso. *Towards Automated Circuit Discovery for Mechanistic Interpretability*. 2023. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997).

[3] C. Olsson et al. *In-context Learning and Induction Heads*. Transformer Circuits Thread, 2022. [Article](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
