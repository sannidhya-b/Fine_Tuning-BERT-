# Fine-Tuning BERT for Sentiment Analysis

**Data Science Internship – February 2026 | Assignment NLP-4**  
Fine-tuning a pre-trained BERT model on IMDb movie reviews for binary sentiment classification.

---

## 🤖 What Is This Project?

This project takes **BERT** — a powerful AI language model pre-trained by Google on billions of words — and teaches it one specific job: **read a movie review and decide if it is Positive or Negative.**

This technique is called **Fine-Tuning** and it is the same approach used by Google, Meta, Amazon, and every modern AI product in the world.

---

## 📂 Repository Structure

```
📦 bert-finetuning-nlp4
 ┣ 📓 bert_finetuning_nlp4.ipynb    ← Main notebook (all experiments + outputs)
 ┗ 📄 README.md                     ← This file
```

---

## 🗂️ Dataset

| Property | Details |
|----------|---------|
| Source | [IMDb Movie Reviews – Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Task | Binary Sentiment Classification |
| Classes | Positive (1) · Negative (0) |
| Built-in sample | 40 reviews — runs without any download |

> To use the full 50,000-review Kaggle dataset, download `IMDB Dataset.csv`,
> upload it to Colab, and uncomment the `pd.read_csv()` line in Cell 2.

---

## 🔁 Pipeline Flow

```
Raw Text
   ↓
Text Preprocessing
(Remove URLs, HTML tags, non-ASCII characters)
   ↓
Tokenization
(bert-base-uncased tokenizer — adds [CLS], [SEP], [PAD] tokens)
   ↓
PyTorch Dataset & DataLoader
(Batches of input_ids + attention_mask + labels)
   ↓
Model Training
(3 Experiments + Bonus DistilBERT)
   ↓
Evaluation
(Accuracy · Precision · Recall · F1 · Confusion Matrix)
   ↓
Comparison & Insights
```

---

## 🧪 Experiments

### Experiment 1 — Freeze All BERT Layers
Only the final classification head is trained. All 12 BERT encoder layers are frozen.

| | |
|---|---|
| **Trainable** | Classification head only |
| **Speed** | Fastest |
| **Best for** | Quick baseline when data is very limited |

---

### Experiment 2 — Fine-Tune Last 2 Layers
The last 2 encoder layers (10 & 11) plus the classifier are trainable. Layers 0–9 are frozen.

| | |
|---|---|
| **Trainable** | Layers 10, 11 + pooler + classifier |
| **Speed** | Medium |
| **Best for** | Good balance of speed and accuracy |

---

### Experiment 3 — Full BERT Fine-Tuning
All 12 encoder layers are trainable — complete fine-tuning on our dataset.

| | |
|---|---|
| **Trainable** | All 12 layers + embeddings + classifier |
| **Speed** | Slowest |
| **Best for** | Maximum accuracy when compute is available |

---

### Bonus — DistilBERT with LR Scheduler + Early Stopping
DistilBERT is a lighter, faster version of BERT with 40% fewer parameters.

| | |
|---|---|
| **Parameters** | ~66M vs BERT's ~110M |
| **Speed** | 60% faster than BERT |
| **Accuracy** | Retains ~97% of BERT performance |
| **Best for** | Production systems where speed matters |

---

## 📊 Expected Results

| Experiment | Trainable Layers | Expected F1 |
|------------|-----------------|-------------|
| Exp 1: Frozen BERT | Classifier only | ~0.65–0.70 |
| Exp 2: Last 2 Layers | Layers 10–11 | ~0.70–0.80 |
| Exp 3: Full Fine-Tuning | All 12 layers | ~0.80–0.90 |
| Bonus: DistilBERT | All (smaller) | ~0.75–0.85 |

> Results improve significantly with the full Kaggle dataset (50,000 reviews).

---

## ⚙️ Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | `bert-base-uncased` |
| Bonus Model | `distilbert-base-uncased` |
| Max Token Length | 128 |
| Batch Size | 8 |
| Epochs | 3 |
| Learning Rate | 2e-5 (AdamW) |
| Scheduler | Linear warmup |
| Early Stopping | Patience = 2 |
| Gradient Clipping | max_norm = 1.0 |

---

## 📈 Evaluation Metrics

- **Accuracy** — Overall correct predictions out of total
- **Precision** — Of all predicted positives, how many were actually positive
- **Recall** — Of all actual positives, how many did the model catch
- **F1 Score** — Harmonic mean of Precision and Recall (main metric)
- **Confusion Matrix** — Visual breakdown of correct vs incorrect predictions per class

---

## 🚀 How to Run


### Jupyter Notebook (Local)
```bash
pip install transformers torch scikit-learn pandas numpy matplotlib seaborn
jupyter notebook bert_finetuning_nlp4.ipynb
```
Then click **Kernel → Restart & Run All**

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `transformers` | BERT and DistilBERT models + tokenizers |
| `torch` | PyTorch — training loop, tensors, GPU support |
| `scikit-learn` | Train/test split, evaluation metrics |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Training curves and bar charts |
| `seaborn` | Confusion matrix heatmaps |

All dependencies are auto-installed in Cell 1 of the notebook.

---

## 💡 Key Concepts Covered

- **Transfer Learning** — Reusing a model pre-trained on billions of words
- **Fine-Tuning** — Adapting a pre-trained model to a specific task
- **BERT Tokenization** — `[CLS]`, `[SEP]`, `[PAD]` tokens, attention masks
- **AdamW Optimizer** — Decoupled weight decay for better regularization
- **Linear Warmup Scheduler** — Gradually increases LR to prevent early instability
- **Early Stopping** — Stops training when validation accuracy stops improving
- **Gradient Clipping** — Prevents exploding gradients during backpropagation

---

## 👤 Author

**Sannidhya Bahulekar**  
Data Science Intern – February 2026  

