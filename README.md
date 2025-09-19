# Efficiency in Multilingual NLP: Knowledge Distillation and Early Exit on XLM-R
Code and experiments for evaluating knowledge distillation and early exit techniques on XLM-R, focusing on performance, efficiency, and cross-linguistic disparities.

## Experimental Process

### 1. Distillation Data Preparation

Sampled subsets of CC100 covering 100 languages are used. The preprocessing involves:

- Filtering CC100 and saving raw text files (per language) into Hugging Face datasets.