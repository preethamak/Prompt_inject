# Documentation Hub

This documentation set is organized as a layered technical report rather than a file dump. The intent is to explain not only what the repository does, but also why the final design was chosen, what results it produced, and what should change next.

## Read in This Order

1. [Research narrative and experiment history](experiments.md)
2. [AutoResearch system architecture](autoresearch-system.md)
3. [Module and method reference](module-reference.md)
4. [Results, limitations, and recommended corrections](results-and-recommendations.md)

## What Each Document Covers

### [experiments.md](experiments.md)

Explains the progression from the weak DeBERTa baseline to embedding-based models, larger-scale training, ensemble attempts, and finally the fused lexical-semantic pipeline.

### [autoresearch-system.md](autoresearch-system.md)

Describes the final reproducible architecture in `autoresearch/`: data loading, branching strategy, fusion logic, threshold calibration, search space design, result persistence, and error mining.

### [module-reference.md](module-reference.md)

Provides a function-by-function and class-by-class reference for the package and a structured summary of the notebooks.

### [results-and-recommendations.md](results-and-recommendations.md)

Interprets the reported metrics, identifies the most meaningful lessons from the experiments, and documents the main technical debts and next corrections.
