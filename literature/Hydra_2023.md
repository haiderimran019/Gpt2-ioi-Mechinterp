# The Hydra Effect: Emergent Self repair in Language Model Computations

Thomas McGrath, Matthew Rahtz, Janos Kramar, Vladimir Mikulik, and Shane Legg, 2023.

Paper: https://arxiv.org/abs/2307.15771

## Why it matters here

The paper shows that ablating one component can change the behavior of other components. This is a reason to treat single head and whole layer scores as marginal intervention effects rather than fixed percentages of an independent circuit.

## Use in this project

The paper supports the caution around zero ablation, cumulative patching, and possible compensation. It does not provide measurements for this repository’s GPT 2 Small IOI run.
