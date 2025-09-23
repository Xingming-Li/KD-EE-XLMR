# Efficiency in Multilingual NLP: Knowledge Distillation and Early Exit on XLM-R
Code and experiments for evaluating knowledge distillation and early exit techniques on XLM-R, focusing on performance, efficiency, and cross-linguistic disparities.

## Experimental Process

### 1. Distillation Data Preparation

Sample and preprocess subsets of CC100 covering 100 languages.

Code (minimal data for a quick access): `./distillation/save_split_data.ipynb`

### 2. Knowledge Distillation

Knowledge distillation from XLM-R Large (teacher) to XLM-R Base (student).

Code: `./distillation/idl.ipynb`

Example distilled model: Hugging Face `xmli/DXLMR-L12M`

### 3. Fine-tuning and Evaluation (Early Exit Integrated)

Fine-tuning and evaluating models with and without early exit.

Code: `./xnli/xnli.ipynb` and `./wikiann/wikiann.ipynb`