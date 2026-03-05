# 🪶 LakotaBERT: Low-Resource Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: RoBERTa](https://img.shields.io/badge/Model-RoBERTa-blue)](https://huggingface.co/docs/transformers/model_doc/roberta)
[![Language: Lakota](https://img.shields.io/badge/Language-Lakota%20(lkt)-green)](https://en.wikipedia.org/wiki/Lakota_language)
🏆 **Best Paper Award at RTIP2P-2024**

**📅 Project Date:** Spring 2024
**🧠 Model Architecture:** RoBERTa (Robustly Optimized BERT)
**📉 Objective:** Masked Language Modeling (MLM) for Endangered Language Revitalization
**🛠️ Tech Stack:** PyTorch, Hugging Face Transformers, Tesseract OCR, Python

---

### 📖 Research Abstract
Lakota is a critically endangered language of the Sioux people in North America. This project introduces **LakotaBERT**, the first large language model (LLM) tailored for Lakota, aiming to support language revitalization efforts.

Unlike English-based models, LakotaBERT was pre-trained from scratch on a custom-compiled corpus of 105K sentences. The model achieved a Masked Language Modeling (MLM) accuracy of 51.48%, demonstrating performance comparable to that of English-based models.

---

### 📊 Performance Metrics
We evaluated the model against baseline models using a single ground truth assumption. The detailed results for LakotaBERT are below:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Accuracy** | **51.48%** | Percentage of masked tokens correctly predicted |
| **Precision** | **0.56** | Proportion of correct predictions among all positive predictions |
| **F1 Score** | **0.49** | Balances precision and recall into a single performance metric |
| **MRR** | **0.51** | Average reciprocal ranks of the correct answers within the predicted lists |
| **CER** | **0.43** | Character-level prediction errors normalized by the length of the longest string |

---

### 🏗️ Pipeline Architecture
The project followed a robust pipeline for training a transformer-based model tailored to Lakota:

* **Data Acquisition:** Gathered datasets from bilingual and monolingual sources, resulting in approximately 105K lines of Lakota and English. Employed the Tesseract OCR engine to extract texts from PDF formats.
* **Tokenization:** Employed Byte Pair Encoding (BPE) during tokenization. Used a vocabulary size of 52,000 to capture the diversity of words and tokens in the Lakota language.
* **Pre-training:** Utilized the RoBERTa architecture with a masking probability of 15% for masked language modeling.

---

### 🚀 Getting Started & Model Weights
The pre-trained model weights, configuration, and tokenizer files are hosted on Hugging Face. You can load the model directly via the `transformers` library:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("kanishka7878/LakotaBERT")
model = AutoModelForMaskedLM.from_pretrained("kanishka7878/LakotaBERT")
```

---

### 💻 Implementation Details
The training script utilizes the Hugging Face `Trainer` API with optimized hyperparameters for low-resource settings:

```python
# Configuration for Low-Resource setting (from src/train_lakota_roberta.py)
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,  # Optimized for smaller dataset size
    type_vocab_size=1,
)
```
