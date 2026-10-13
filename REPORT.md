# Robustness of Attention Head Intervention Rankings in GPT 2 Small

## An evidence based study of the Indirect Object Identification task

### Author

Haider Imran, independent researcher

### Author note

This report is written by Haider Imran as an independent research project in mechanistic interpretability. The goal is to explain the work clearly and honestly. The study is based on the outputs saved inside the original notebook in this repository. It is not presented as a new run performed during this documentation update. The original notebook, its printed results, and its embedded figures are the primary evidence.

## Abstract

Large language models often produce correct answers without showing us how they reached those answers. Mechanistic interpretability tries to study that hidden process by examining internal model components and testing them with controlled interventions. This paper studies the Indirect Object Identification task in GPT 2 Small. In this task, the model must choose the name that is not the subject of the final clause. For example, in the sentence “When Mary and John went to the store, John gave a drink to”, the expected answer is Mary.

The original notebook evaluated one main prompt set with 100 examples and two held out prompt sets. The first held out set changed the wording. The second used different names, places, and objects. The notebook reported a clean logit difference of 3.60 on the main set and 100 percent pairwise accuracy. Corrupted activation patching ranked L8H6 as the strongest head, with a score of 0.379 and a bootstrap 95 percent interval from 0.350 to 0.409. The ranking was similar on the held out wording set, with Spearman correlation 0.933 and ten of ten top heads shared. It was also similar on the held out vocabulary set, with correlation 0.912 and ten of ten top heads shared.

The notebook also tested complete MLP outputs. MLP layer 0 and MLP layer 5 had large effects under these interventions. However, the study did not perform a factorial head and MLP experiment or path patching. Therefore, it does not show that a particular MLP carries the effect of a particular attention head. The most careful conclusion is that the stored experiment gives evidence for stable head rankings within a small controlled IOI setting. It does not claim a new circuit or a new head and MLP mechanism.

The top ten heads also line up with roles described in the published IOI circuit work. L8H6, L8H10, L7H9, and L7H3 are commonly described as S inhibition heads. L5H5 and L6H9 are induction related heads, L3H0 is a duplicate token head, and L9H9 and L10H0 are name mover heads. This is a comparison with a published circuit, not a new circuit discovery. The notebook also recorded negative patching scores for L10H7 at negative 0.483 and L11H10 at negative 0.251. In this metric, those scores mean that patching the head increased the clean IO over subject gap. This sign is consistent with negative name mover behavior, but the stored run did not reproduce the full causal analysis from the published work.

## 1. Introduction

### 1.1 Why study the inside of a language model

Modern language models can complete sentences, answer questions, and follow instructions. Their behavior can look understandable from the outside, but the internal process is usually difficult to inspect. A model can select the correct answer for reasons that are very different from the reasons a human would give.

Mechanistic interpretability is an effort to reduce this uncertainty. Instead of treating the model as one large black box, researchers inspect smaller parts such as attention heads, MLP layers, neurons, and residual stream representations. They then ask whether changing one part changes a behavior. This approach is useful because it connects a behavioral result with a testable internal explanation.

The purpose of an intervention is to create a controlled counterfactual. We can ask what the model would do if an internal activation were replaced with an activation from another example, or if an internal output were set to zero. An intervention does not automatically reveal the complete mechanism. It gives evidence about the component under the particular intervention, dataset, and metric that were chosen.

### 1.2 The Indirect Object Identification task

Indirect Object Identification, often shortened to IOI, is a simple language task that has become an important example in mechanistic interpretability. Consider this sentence:

> When Mary and John went to the store, John gave a drink to

The model should predict Mary. Mary is the indirect object because John is the subject of the final clause and Mary is the other name introduced earlier.

This task is useful for a small study because the correct answer is known, the prompts can be generated in a controlled way, and the answer can be measured with a direct comparison between two name logits. It also allows us to ask a meaningful internal question. Which model components help the model prefer Mary over John?

### 1.3 Aim of this paper

The original project started as a learning and replication exercise. It reproduced several standard mechanistic interpretability procedures, including attention head patching, zero ablation, cumulative patching, bootstrap intervals, attention visualisation, and MLP interventions.

The question in this paper is deliberately modest:

> Are attention head intervention rankings stable when the wording and vocabulary of the IOI prompts are changed, and what can the existing MLP results tell us about a possible interaction between attention heads and MLP layers?

The first part can be answered by the stored notebook outputs. The second part cannot be answered fully because a direct head and MLP interaction experiment was not run. Reporting that boundary is part of the result.

## 2. Prior work

### 2.1 The IOI circuit in GPT 2 Small

Wang, Variengien, Conmy, Shlegeris, and Steinhardt studied IOI in GPT 2 Small in their paper “Interpretability in the Wild” [1]. They did not stop at measuring whether the model selected the correct name. They investigated how information moves through the network and described a circuit containing multiple groups of attention heads. Their work identified 26 important heads organized into seven broad functional groups.

The groups included heads that move information about names, heads that identify the repeated subject, heads that provide positional or contextual information, and heads that help produce the final answer. The important lesson for this project is that an IOI explanation is not simply a list of heads with high scores. A strong explanation must connect components to a proposed computation and test the proposal with interventions.

The current project uses the same general task and model family, but it does not claim to rediscover or replace the circuit in Wang et al. The present study is narrower. It asks whether the ranking from a simple head intervention remains similar under two controlled changes to the prompts.

### 2.2 Automated circuit discovery

Conmy and colleagues introduced ACDC, or Automatic Circuit DisCovery, in “Towards Automated Circuit Discovery for Mechanistic Interpretability” [2]. Their work focuses on the connections between components rather than only on the components themselves. This distinction is important. If a head and an MLP both matter, that does not tell us whether the head sends useful information to that MLP, whether both act independently, or whether the model can compensate when one is removed.

ACDC also shows why a circuit result depends on choices made by the researcher. The task, dataset, metric, intervention method, and threshold can all influence the circuit that is discovered. The current project therefore records its prompt templates, seeds, metric, and intervention descriptions instead of presenting the ranking as a universal property of GPT 2 Small.

### 2.3 Interactions and non additive effects

Several lines of work explain why single component interventions can be difficult to interpret. The Hydra Effect describes compensation after ablation, including cases where other attention layers change their behavior [3]. Practical guidance on activation patching also warns that the choice of metric, corruption, and patch location affects what a score means [4]. The published IOI study additionally describes backup heads that can become important when primary heads are disrupted [1].

These ideas matter for the present MLP results. Removing or replacing one component can change the activation distribution seen by later components. As a result, the effect of removing two components together may be larger or smaller than the sum of their separate effects.

This point motivates the next experiment for this project. A complete head and MLP study must test the two components alone and together. The existing notebook does not contain those four condition comparisons for any selected head and MLP pair.

### 2.4 How this project fits the literature

The project is best described as an evidence based robustness study connected to the IOI circuit literature. It is not an independent claim that a new circuit has been discovered. It is also not an automated circuit discovery method. Its value is more limited and more practical. It records whether a beginner’s component ranking remains similar across two small prompt changes and makes the limits of that conclusion visible.

## 3. Research question and expectations

### 3.1 Research question

The main research question is:

> When GPT 2 Small is evaluated on controlled IOI prompts, do attention head rankings from corrupted activation patching remain stable across a held out wording template and a held out vocabulary set?

A secondary question is:

> Do the existing whole MLP intervention results provide enough evidence to claim that a specific attention head works through a specific MLP layer?

### 3.2 Expectations before reading the results

The first expectation was that some heads would be more important than others, because earlier IOI research found a structured circuit [1]. The second expectation was that a ranking based only on one exact sentence template might change when the wording or vocabulary changed. The third expectation was that whole MLP interventions could reveal important layers, but would not by themselves prove a route from a particular head to a particular MLP.

## 4. Methods

### 4.1 Model and recorded environment

The notebook records GPT 2 Small with 12 transformer layers and 12 attention heads in each layer. It records Python 3.13.15, PyTorch 2.11.0 with CUDA 12.8, TransformerLens 4.0.0, Transformers 5.17.0, and a Tesla T4 GPU.

These details come from the notebook output. The current repository does not contain the original machine image, activation caches, or a dependency lock file. For that reason, this paper calls the measurements stored notebook results rather than a fresh independent replication.

### 4.2 Prompt sets

The main prompt set is called D0. It contains 100 examples generated with seed 0. The main template was:

> When {first} and {second} went to the {place}, {subject} gave a {object} to

The notebook created clean and corrupted versions of each prompt. The clean version preserves the intended subject and indirect object relationship. In the recorded D0 example, the clean prompt ends with “Jack gave a note to” and the corrupted prompt replaces the subject name with a third name, “Mary”. This is a structured counterfactual, not random noise. It changes the subject role while keeping the template and most of the vocabulary fixed.

The first held out set is called D1. It contains 100 examples generated with seed 1 and a different wording template:

> After {first} and {second} arrived at the {place}, {subject} handed a {object} to

The second held out set is called D2. It contains 100 examples generated with seed 2. It uses the original style of sentence but uses a separate list of names, places, and objects. The purpose of D2 is to check whether the ranking is tied too closely to the exact words in D0.

### 4.3 Behavioral metric

For each prompt, the notebook measured the logit for the correct indirect object name and subtracted the logit for the subject name. This value is called the logit difference. A positive value means that the model prefers the indirect object. A negative value means that it prefers the subject.

The notebook also reported pairwise accuracy. A prompt counted as correct when the logit for the indirect object was greater than the logit for the subject.

### 4.4 Attention head patching

For each of the 144 attention heads, the notebook replaced the clean activation with the corresponding activation from the corrupted prompt. It then measured the new logit difference. The patching score was calculated as the fraction of the clean to corrupted mean gap removed by the intervention.

This score is useful for ranking heads, but it should not be interpreted as a literal percentage of a complete circuit. Scores can exceed one because model components interact and because the normalized metric compares means rather than independent additive contributions.

### 4.5 Zero ablation

The notebook also set each head output to zero one at a time. This tests a different counterfactual from corrupted activation patching. Patching gives a component a signal from a matched corrupted example. Zero ablation removes the signal entirely. Agreement between these methods would strengthen confidence that a ranking is not specific to one intervention. Disagreement is also informative because it shows that the methods measure different aspects of the model.

### 4.6 Bootstrap intervals

The notebook calculated bootstrap 95 percent intervals for the strongest D0 patching scores. The interval estimates the variation produced by resampling the prompt examples. The original prompt level values are not present as a separate file in this checkout, so the intervals are preserved as reported results and cannot be independently recalculated here.

### 4.7 Cumulative patching

The notebook selected the highest ranked heads and patched them together in groups of increasing size. It compared these selected groups with randomly chosen groups. This asks whether the ranking concentrates useful intervention effects into a small set of heads.

### 4.8 Whole MLP interventions

The notebook repeated the patching and zero ablation idea for the post activation of each MLP layer. These interventions affect the complete MLP output at a layer. They can therefore show that a layer has an important total effect, but they do not identify individual neurons or a specific incoming head to MLP route.

### 4.9 Evidence preservation

The original notebook is preserved without edits. The new script `scripts/extract_notebook_evidence.py` copies the printed numerical record into JSON and CSV files and extracts the four non empty figures embedded in the notebook. The script does not load a model and does not create new measurements.

## 5. Results

### 5.1 Basic behavior

The notebook reported the following values.

| Dataset | Description | Clean logit difference | Corrupted logit difference | Accuracy |
|---|---|---:|---:|---:|
| D0 | Main template, 100 prompts | 3.60 | negative 0.32 | 100 percent |
| D1 | Held out wording, 100 prompts | 3.42 | negative 0.19 | 100 percent |
| D2 | Held out vocabulary, 100 prompts | 3.88 | negative 0.23 | 99 percent |

The clean values were positive in all three sets. The corrupted values were negative in all three sets. This means that the notebook’s prompt construction produced the expected contrast between the correct indirect object and the competing subject.

### 5.2 Main attention head ranking

The strongest D0 patching score was obtained by L8H6, with a score of 0.379. Its stored bootstrap interval was [0.350, 0.409]. The next strongest heads were L5H5 with 0.281, L8H10 with 0.279, L7H9 with 0.239, and L9H9 with 0.218.

The direct comparison between patching and zero ablation produced a Spearman correlation of 0.485 across all 144 heads. Six heads appeared in both top ten lists: L8H6, L8H10, L7H9, L9H9, L6H9, and L5H9.

The zero ablation ranking also exposed an important off distribution effect. L0H9 had a zero ablation score of 0.935, much larger than its corrupted patching score. This head is not in the patching top ten. The difference is a reminder that zeroing a component is a stronger distribution shift than replacing it with a matched corrupted activation.

The notebook recorded five heads for which patching increased the clean IO over subject gap rather than reducing it: L10H7 at negative 0.483, L11H10 at negative 0.251, L11H2 at negative 0.074, L9H6 at negative 0.064, and L9H4 at negative 0.018. The first two are consistent with the negative name mover role described in prior IOI work.

The moderate rather than perfect agreement is important. It means that “importance” depends partly on the intervention used. A head can have a large effect when its activation is replaced by a matched corrupted activation and a smaller effect when its output is simply removed.

### 5.3 Held out wording

The D1 patching ranking had a Spearman correlation of 0.933 with the D0 ranking. Ten of the ten top heads overlapped. The notebook reported 100 percent pairwise accuracy on D1, with clean logit difference 3.42 and corrupted logit difference negative 0.19.

This result suggests that the ranking was not limited to one exact surface form. However, the two templates were both short, artificial sentences with the same basic relation between two names. This is a controlled robustness check, not evidence that the ranking will remain unchanged on natural language.

### 5.4 Held out vocabulary

The D2 patching ranking had a Spearman correlation of 0.912 with the D0 ranking. Ten of the ten top heads overlapped. The notebook reported 99 percent pairwise accuracy on D2, with clean logit difference 3.88 and corrupted logit difference negative 0.23.

This result suggests that the ranking did not depend entirely on the exact names, places, and objects used in D0. The lexical list was still small and curated, so the correct interpretation is limited. It is evidence of stability across this vocabulary change, not a general claim about all words.

### 5.5 Cumulative patching

The notebook reported that the top one head removed 0.38 of the mean gap, compared with 0.00 for a random head group. At five heads, the top group removed 0.83 compared with 0.10 for the random group. At eight heads, the top group removed 1.00 compared with 0.09 for the random group. At ten heads the reported top group value fell to 0.97, so the curve was not monotonic.

The selected heads therefore concentrated much more of the measured intervention effect than the notebook’s random baseline. However, this comparison is circular because the heads were selected and scored on D0. The random groups were not layer matched, and the top groups were not scored on D1 or D2 in the stored outputs. Values close to or above one at larger group sizes should not be read as proof that the selected heads form a complete additive circuit. They reflect the behavior of a normalized intervention metric in a network with interacting components.

### 5.6 MLP results

The whole MLP patching ranking was MLP0 at 0.973, MLP5 at 0.224, MLP4 at 0.131, MLP2 at 0.095, and MLP1 at 0.082. The whole MLP zero ablation ranking was MLP0 at 1.253, MLP5 at 0.321, MLP7 at 0.246, MLP6 at 0.214, and MLP4 at 0.209.

These values show that whole MLP outputs can have large effects on the chosen logit difference, but they need careful interpretation. MLP0 is the first MLP after the token embedding and can partly re encode token identity. Its high patching score may therefore reflect the corruption passing through an early representation rather than a task specific IOI mechanism. A zero ablation score above one, such as MLP0 at 1.253, means that the intervention moved the metric beyond the corrupted reference. It is not a percentage of a circuit. Finally, each MLP intervention changes the full MLP output, whereas a head intervention changes one 64 dimensional head output. These interventions are not matched in granularity.

The stored results also do not answer which heads provide information to an MLP, whether the MLP acts before or after a head’s contribution, or whether another component compensates for the intervention.

## 6. Discussion

### 6.1 What the results mean in simple terms

The strongest simple conclusion is that the same small group of attention heads remained important when the prompt wording and vocabulary were changed in the two ways tested by the notebook. The model’s behavior was also stable at the basic level. It preferred the correct indirect object on nearly every example.

In ordinary language, this means that the result did not appear to be caused only by one exact sentence or one small list of names. The model seems to use internal signals that are useful for the relation being tested.

The results do not tell us that the model has one small independent circuit in which every head contributes a fixed percentage. The moderate agreement between patching and zero ablation shows that the intervention itself matters. The MLP results show that broad layer interventions matter too, but they do not identify the path by which information travels.

### 6.2 Why the MLP result is interesting

The MLP finding is a useful starting point for a stronger follow up study. A beginner might look at a strong L8H6 head score and a strong MLP5 score and conclude that L8H6 must send its information through MLP5. That is not only untested but also has the causal direction backwards. In a GPT 2 block, attention is computed before that block’s MLP. An L8H6 output can directly influence MLP8 or later through the residual stream, but it cannot send information backwards into MLP5. MLP5 could influence a later head, or both components could matter independently. The apparent effect could also involve a different residual stream route or compensation by other heads.

The correct next test is a factorial intervention. For one selected head and one selected MLP, measure four conditions on the same prompts. The first condition leaves both components unchanged. The second intervenes on the head only. The third intervenes on the MLP only. The fourth intervenes on both. The combined effect can then be compared with the sum of the two separate effects.

If the combined effect is reliably larger or smaller than the additive expectation, that would be evidence of an interaction. It would still not fully explain the mechanism. Path patching and additional controls would be needed to determine the direction of information flow.

### 6.3 Relationship to prior IOI research

The result is consistent with the broader picture from Wang et al. [1], where multiple heads with different roles contribute to IOI. The present top ten can be mapped cautiously onto that published picture: L8H6, L8H10, L7H9, and L7H3 match the S inhibition group; L5H5 and L6H9 match induction related support; L3H0 matches duplicate token detection; and L9H9 and L10H0 match name mover roles. The negative scores for L10H7 and L11H10 also agree in sign with the published negative name mover group. This is a role comparison, not a replacement for the published circuit analysis. Wang and colleagues used a wider set of causal tests and evaluated their explanation using faithfulness, completeness, and minimality.

The result also fits the warning from ACDC [2] and related work [3, 4]. Finding components with large marginal effects is an important step, but a circuit explanation requires studying connections and interactions. The current project has completed the component ranking stage and is ready for a carefully designed interaction study.

## 7. Limitations

### 7.1 Stored output rather than independent rerun

The largest limitation is evidence provenance. The numerical values come from the original notebook’s printed outputs. The current checkout does not include raw activation caches, per prompt intervention values, or the old result tensor files named in earlier documentation. The new JSON and CSV files are a transparent transcription of the notebook output. They are not newly generated model measurements.

### 7.2 Small and artificial prompt distribution

Each dataset contains 100 controlled prompts. The templates are short, and the lexical lists are curated. This is appropriate for a first study because it reduces unwanted variation. It also limits the scope of the result. The paper cannot claim that the ranking holds for ordinary prose, other languages, other tasks, or larger models.

### 7.3 One model and one recorded run

Only GPT 2 Small is represented. The notebook records one seed for each dataset. The results do not measure variation across independently loaded model copies, training checkpoints, random seeds, or model sizes.

### 7.4 Marginal interventions are not a complete circuit

The head scores are marginal effects. They do not reveal all important edges. They can also be affected by redundancy, layer normalisation, downstream compensation, and changes in activation statistics. Cumulative patching is useful for concentration, but it does not prove that the selected heads are a complete minimal circuit.

### 7.5 Ranking uncertainty and held out evaluation

The stored notebook gives bootstrap intervals for several D0 head scores, but it does not give confidence intervals for the Spearman correlations, uncertainty for top ten overlap, or a same template new seed noise ceiling. It also does not evaluate the D0 selected top k sets on D1 and D2. Therefore, the reported stability is descriptive. A stronger study would predefine an instability threshold, report rank uncertainty, use top k agreement and scatter plots, and evaluate selected sets on held out data.

### 7.6 No direct head and MLP interaction test

The central uncompleted experiment is a joint intervention design. Since it was not run, the paper does not claim that any particular head works through any particular MLP layer. This limitation is not a failure of the project. It identifies a precise next experiment.

### 7.7 Historical documentation errors

The original repository contained conflicting prose about the D2 correlation and overlap. One notebook prose cell reported 0.781 and nine of ten, while the direct printed output reported 0.912 and ten of ten. The notebook summary has now been corrected to match the direct printed output. This paper uses the direct printed output and records the correction openly. It also avoids claiming that absent result files exist.

## 8. Reproducibility and project files

The original notebook remains the historical source. The manifest in `configs/executed_notebook_manifest.json` records the model, data splits, metric, interventions, and known evidence limits. The extractor script creates `results/executed_notebook_evidence.json`, `results/baseline_summary.csv`, and `results/head_patching_summary.csv`. It also extracts the four figures embedded in the notebook.

The README explains how a reader can materialise this evidence without downloading a model. A future replication should preserve every prompt, every per prompt logit difference, every intervention value, the full software environment, and the random seeds. It should also save the exact model identifier and checksum. The generated PDF is stored at `output/pdf/gpt2_ioi_research_paper.pdf` and is built from this report by `scripts/build_paper_pdf.py`.

## 9. Conclusion

This project provides a clear and limited result. In the stored GPT 2 Small IOI experiment, corrupted activation patching produced a stable attention head ranking across one held out wording template and one held out vocabulary set. The main D0 result identified L8H6 as the strongest head, and the top ten overlap was complete in both held out comparisons.

The experiment also showed that whole MLP interventions can strongly affect the same metric. That observation motivates a head and MLP interaction study, but it does not establish one. The honest contribution of this paper is therefore a documented robustness and replication result with a clear boundary between measured evidence and planned work.

The next research decision is to choose whether to submit this as a replication and audit report or to run and document the factorial interaction experiment first. Either choice can be scientifically useful. The important requirement is that the final paper should say exactly what was measured, what was inferred, and what remains unknown.

## Acknowledgements and credit

This project builds on the IOI task and mechanistic interpretability methods developed by earlier researchers. Special credit goes to Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt for the original GPT 2 Small IOI circuit study. Credit also goes to Arthur Conmy and his coauthors for ACDC, and to the Transformer Circuits authors for the discussion of induction heads and interactions in models with MLPs.

The software used by the original notebook included TransformerLens and Hugging Face Transformers. The current repository preserves the original notebook rather than claiming that the present documentation update independently reproduced its execution.

## References

[1] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. “Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT 2 Small.” 2022. [https://arxiv.org/abs/2211.00593](https://arxiv.org/abs/2211.00593)

[2] Arthur Conmy, Augustine Mavor Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga Alonso. “Towards Automated Circuit Discovery for Mechanistic Interpretability.” 2023. [https://arxiv.org/abs/2304.14997](https://arxiv.org/abs/2304.14997)

[3] Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg. “The Hydra Effect: Emergent Self repair in Language Model Computations.” 2023. [https://arxiv.org/abs/2307.15771](https://arxiv.org/abs/2307.15771)

[4] Stefan Heimersheim and Neel Nanda. “How to use and interpret activation patching.” 2024. [https://arxiv.org/abs/2404.15255](https://arxiv.org/abs/2404.15255)

[5] Christopher Olsson and colleagues. “In Context Learning and Induction Heads.” Transformer Circuits Thread, 2022. [https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)

[6] TransformerLens documentation and source project. “TransformerLens.” [https://github.com/TransformerLensOrg/TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)

[7] Hugging Face. “Transformers documentation.” [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
