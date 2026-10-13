# Literature Review

This literature review is divided into two related areas: **mechanistic interpretability** and **relational learning**.

The first two papers focus on understanding and discovering internal computational circuits in neural networks. The third explores a different but related question: how neural networks can learn from structured relationships in data.

---

## Mechanistic Interpretability

### Wang et al. (2022)

**Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small**

Focus:
- Indirect Object Identification (IOI)
- Circuit discovery
- Attention heads
- Activation/path patching
- Causal interventions

**Main question:**  
How does GPT-2 Small internally perform IOI?

[My notes →](./IOI_2022_Wang.md)

---

### Conmy et al. (2023)

**Towards Automated Circuit Discovery for Mechanistic Interpretability**

Focus:
- Automated circuit discovery
- ACDC
- Causal interventions
- Computational graphs
- Circuit faithfulness

**Main question:**  
Can the process of discovering neural circuits be partially automated?

[My notes →](./ACDC_2023.md)

### McGrath et al. (2023)

**The Hydra Effect: Emergent Self repair in Language Model Computations**

Focus:
- Compensation after ablation
- Interacting attention layers
- Interpretation limits of single component interventions

**Main question:**
Can other components change their behavior after one component is ablated?

[Paper →](https://arxiv.org/abs/2307.15771)

### Heimersheim and Nanda (2024)

**How to use and interpret activation patching**

Focus:
- Activation patching design
- Metric and corruption choices
- Interpreting causal evidence carefully

**Main question:**
What can activation patching scores support, and what common conclusions go beyond the evidence?

[Paper →](https://arxiv.org/abs/2404.15255)

---

## Related Work

### Fey et al. (2024)

**Position: Relational Deep Learning - Graph Representation Learning on Relational Databases**

Focus:
- Relational databases
- Graph representation learning
- Heterogeneous and temporal graphs
- Graph Neural Networks
- RelBench

**Main question:**  
Can neural networks learn directly from the structure of relational databases without requiring extensive manual feature engineering?

[My notes →](./relational-deep-learning.md)

---

## How These Papers Connect

My reading started with mechanistic interpretability:

```text
Wang et al.
     ↓
Understanding a circuit inside GPT-2
     ↓
Conmy et al.
     ↓
Automating circuit discovery
```

I then became interested in another way that **structure** appears in machine learning:

```text
Relational Deep Learning
     ↓
Learning from relationships between entities
     ↓
Graph representation learning
```

These are not the same research problem. I am keeping the relational learning paper as **related work** because it introduces a different perspective on learning from structure and is relevant to the research direction I am exploring.

---

## Current Understanding

The main idea I have taken from these papers is that understanding structure is important at several levels:

- **Inside a model:** Wang et al. investigate how components interact to produce a behavior.
- **Discovering model structure:** Conmy et al. investigate how those circuits can be found more automatically.
- **In the data:** Relational Deep Learning investigates how models can learn from relationships between entities.

I am still exploring how, if at all, these perspectives can be connected into a more specific research question.
