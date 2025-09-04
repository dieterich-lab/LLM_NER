# LLM\_NER — German Clinical NER with Surrogate vs. Original Corpora

## TL;DR

This project evaluates whether a **surrogate (privacy-preserving) corpus** can replace an **original** corpus for **Named Entity Recognition (NER)** on German clinical texts **without degrading downstream utility**.
Result: **No degradation** observed — F1 on surrogate is slightly **higher** than on original.

| Corpus    | TP  | FP  | FN  | Precision | Recall | F1     |
| --------- | --- | --- | --- | --------- | ------ | ------ |
| Surrogate | 783 | 147 | 409 | 0.8419    | 0.6569 | 0.7380 |
| Original  | 786 | 167 | 414 | 0.8248    | 0.6550 | 0.7301 |

## What’s in here

* `data/`

  * `original_txt/`, `original_json/`, `original_ann/`
  * `fictive_txt/`, `fictive_json/`, `fictive_ann/` (surrogate)
* `LLM_output/openai/gpt-oss-20b/{original,fictive}/` — model predictions (JSON)
* `NER_eval.ipynb` — evaluation notebook (corpus-wide + per-class metrics, confusion matrix, plots)
* `LLM_NER.ipynb` — inference notebook that:

  * loads **openai/gpt-oss-20b** (HF Transformers)
  * prompts for **German clinical NER** with a De-ID label set (e.g., NAME\_PATIENT, DATE\_BIRTH, LOCATION\_*, CONTACT\_*, ID, etc.)
  * writes prediction JSONs alongside Pydantic validation
* `eval_results/original/*.png` — sample plots (confusion matrix, per-class metrics)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Quickstart

### 1) Generate predictions with the LLM

Open **`LLM_NER.ipynb`** and run the cells:

* Choose the input folder:

  * `data/original_txt` **or** `data/fictive_txt`
* The notebook builds a chat-style prompt for **German clinical De-ID NER** and saves model outputs as JSON to:

  * `LLM_output/openai/gpt-oss-20b/{original|fictive}/`

Each JSON maps **exact text spans → labels** and is validated with **Pydantic** to keep labels consistent.

### 2) Evaluate

Open **`NER_eval.ipynb`** and run:

* Compares **generated JSON** vs. **gold JSON** (`data/*_json`)
* Reports:

  * overall **Precision / Recall / F1**
  * **per-class** metrics
  * **confusion matrix** + **per-class bar chart**
* Saves figures to `eval_results/`

### 3) Reproducing headline numbers

Use:

* Predictions: `LLM_output/openai/gpt-oss-20b/{fictive,original}/`
* Gold: `data/{fictive_json,original_json}/`

The corpus-wide summary (already computed) is:

| Corpus    | TP  | FP  | FN  | Precision | Recall | F1     |
| --------- | --- | --- | --- | --------- | ------ | ------ |
| Surrogate | 783 | 147 | 409 | 0.8419    | 0.6569 | 0.7380 |
| Original  | 786 | 167 | 414 | 0.8248    | 0.6550 | 0.7301 |
