# How to use and interpret activation patching

Stefan Heimersheim and Neel Nanda, 2024.

Paper: https://arxiv.org/abs/2404.15255

## Why it matters here

Activation patching results depend on the corruption, the location patched, and the metric used to measure the output. A large score is evidence about the chosen counterfactual, not automatically a complete circuit explanation.

## Use in this project

This guidance supports reporting the stored head scores as descriptive intervention rankings. It also motivates future rank uncertainty, held out top k evaluation, and path patching. It does not change any stored measurement.
