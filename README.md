# Efficiency in Multilingual NLP: Knowledge Distillation and Early Exit on XLM-R
Code and experiments for evaluating knowledge distillation and early exit techniques on XLM-R, focusing on performance, efficiency, and cross-linguistic disparities.

## Experimental Process

### 1. Distillation Data Preparation

Sampled subsets of CC100 covering 100 languages are used. The preprocessing involves:

- Filtering CC100 and saving raw text files per language.
- Tokenizing with XLM-R tokenizer.
- Saving tokenized data to compressed Parquet files.
- Splitting each language dataset into 5 equal-sized chunks for later distillation.

Code (minimal data for a quick access): `./distillation/save_split_data.ipynb`

### 2. Knowledge Distillation

Knowledge distillation from XLM-R Large (teacher) to XLM-R Base (student):

- Loading tokenized and split datasets for selected languages.
- Training the student model with distillation loss, combining MSE loss between intermediate layers and KL divergence between final logits.

