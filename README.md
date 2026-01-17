# 🌙 Arabic NLP Classification CLI

<div align="center">

### *From Raw Arabic Text to Production-Ready Models in One Command*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CLI](https://img.shields.io/badge/CLI-Tool-green.svg)](https://github.com/yourusername/arabic-nlp-cli)
[![Arabic](https://img.shields.io/badge/Language-Arabic-orange.svg)](https://en.wikipedia.org/wiki/Arabic)

</div>


## 🎯 The Problem

You have Arabic text data. You need a classification model. Between you and production are:
- Data exploration notebooks
- Preprocessing scripts with encoding nightmares
- Embedding experiments across multiple files
- Training code scattered everywhere
- Performance tracking in random cells

**Days of work. Dozens of files. One headache.**

## ✨ The Solution

```bash
# One command. One pipeline. Done.
python main.py pipeline reviews.csv review_text rating --embed model2vec

```
**5 minutes later:**
- ✅ Exploratory visualizations generated
- ✅ 40,000+ texts preprocessed and normalized
- ✅ Semantic embeddings created
- ✅ 4 models trained and evaluated
- ✅ Best model saved with full performance report

---

## 🚀 What Makes This Special

<table>
<tr>
<td width="50%">

### 🧠 **Smart Arabic Processing**
Built specifically for Arabic's complexity:
- Handles diacritics, elongation, letter variants
- Context-aware stopword removal
- ISRI stemming for morphological richness
- Zero encoding issues

</td>
<td width="50%">

### ⚡ **Blazing Fast Workflow**
No more context switching:
- Single command = complete pipeline
- Automatic intermediate file handling
- Progress tracking at every step
- Resumable from any checkpoint

</td>
</tr>
<tr>
<td width="50%">

### 📊 **Production-Ready Outputs**
Everything you need to ship:
- Markdown reports with metrics
- classification reports
- Saved models ready for deployment
- Beautiful visualizations for stakeholders

</td>
<td width="50%">

### 🎨 **Visual Insights**
Understand your data instantly:
- Class distribution analysis
- Text length patterns
- Top word frequencies
- Comparative model performance

</td>
</tr>
</table>

---


## 🌊 The Pipeline Flow
```
📁 Your CSV
    ↓
🔍 Data Validation
    ↓
📊 Visual EDA
    ↓
🧹 Arabic Preprocessing
    ↓
🧠 Embedding Choice
    ├─→ ⚡ TF-IDF Vectors
    └─→ 🧠 Model2Vec ARBERTv2
         ↓
    🎓 Multi-Model Training
         ↓
    📈 Performance Reports
         ↓
    ⭐ Best Model Selection
         ↓
    🎉 Production Ready!
```

---

## 🛠️ Installation
### Setup in 3 Steps

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# 3. Sync dependencies with uv
uv sync
# 4 Make sure you are in Arabic_NLP_Project_CLi
cd Arabic_NLP_Project_CLi
```



### Verify Installation

```bash
python main.py --help
```

**Expected output:**
```text
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  Arabic NLP CLI Tool - End-to-end pipeline for Arabic text classification.

Options:
  --help  Show this message and exit.

Commands:
  pipeline  Run full pipeline: Load → EDA → Preprocess → Embed → Train
```

✅ **You're ready to go!**

---

## 🎬 Quick Start

### The One-Liner

```bash
python main.py pipeline CompanyReviews.csv review_description rating --embed model2vec
```

**What happens:**

<details>
<summary><b>📥 Step 1: Data Loading & Validation</b></summary>

```text
Step 1: Loading and validating data...
✓ Loaded: 40,046 rows, 4 columns
✓ Missing text rows: 1
✓ Number of classes: 3
✓ Step 1 finished successfully
```

The tool validates your CSV, checks for missing values, and confirms class distribution.

</details>

<details>
<summary><b>📊 Step 2: Exploratory Data Analysis</b></summary>

```text
Step 2: Running EDA...
✓ Saved pie chart: outputs/visualizations/eda_class_distribution_pie.png
✓ Saved words histogram: outputs/visualizations/eda_text_length_words.png
✓ Saved chars histogram: outputs/visualizations/eda_text_length_chars.png
✓ Saved top words chart: outputs/visualizations/eda_top_words.png
✓ Step 2 finished successfully
```

Generates 4 publication-ready visualizations automatically.

</details>

<details>
<summary><b>🧹 Step 3: Arabic Text Preprocessing</b></summary>

Your text transforms through:
- Diacritic removal: `مُحَمَّد` → `محمد`
- Letter normalization: `إسلام` → `اسلام`
- Elongation handling: `وااااو` → `واو`
- Stopword filtering
- ISRI stemming

**Result:** Clean, standardized Arabic ready for ML.

</details>

<details>
<summary><b>🧠 Step 4: Semantic Embeddings</b></summary>

```text
Step 4: Creating embeddings...
🧠 Model2Vec shape: (40,046, 128)
💾 Saved embeddings: outputs/embeddings/model2vec_embeddings.npy
✅ Embeddings created successfully
```

Uses pre-trained ARBERTv2 for context-aware Arabic representations.

</details>

<details>
<summary><b>🎓 Step 5: Model Training & Evaluation</b></summary>

```text
Step 5: Training and reporting...
📝 Saved report: outputs/reports/training_report_2026-01-17_14-32-08.md
⭐ Best model: Random Forest (accuracy=0.7861)
💾 Saved best model: outputs/models/best_model_model2vec.pkl
✅ Training completed successfully
```

Trains Logistic Regression, Random Forest, SVM, and Gradient Boosting simultaneously.

</details>

---

## 🔬 Under the Hood

### Preprocessing Pipeline

<table>
<thead>
<tr>
<th width="20%">Step</th>
<th width="40%">What It Does</th>
<th width="40%">Example</th>
</tr>
</thead>
<tbody>
<tr>
<td>🔤 <b>Lowercasing</b></td>
<td>Standardize text case</td>
<td><code>النص الجميل</code> → <code>النص الجميل</code></td>
</tr>
<tr>
<td>✨ <b>Diacritic Removal</b></td>
<td>Strip Tashkeel marks (ً ٌ ٍ َ ُ ِ ّ ْ)</td>
<td><code>هُوَ جَمِيلٌ</code> → <code>هو جميل</code></td>
</tr>
<tr>
<td>🔄 <b>Letter Normalization</b></td>
<td>Unify letter variants</td>
<td><code>إسلام، أمل، آية</code> → <code>اسلام، امل، اية</code><br><code>ى → ي, ة → ه, ؤ → و</code></td>
</tr>
<tr>
<td>➖ <b>Elongation Removal</b></td>
<td>Collapse repeated characters</td>
<td><code>رااااائع جدااااا</code> → <code>رائع جدا</code></td>
</tr>
<tr>
<td>🧹 <b>Text Cleaning</b></td>
<td>Remove noise (numbers, punctuation, extra spaces)</td>
<td><code>السعر 500 ريال!!!</code> → <code>السعر ريال</code></td>
</tr>
<tr>
<td>🚫 <b>Stopword Filtering</b></td>
<td>Remove common words from <code>arabic_stopwords.txt</code></td>
<td><code>هذا هو النص من المقال</code> → <code>النص المقال</code></td>
</tr>
<tr>
<td>🌱 <b>ISRI Stemming</b></td>
<td>Extract word roots</td>
<td><code>يكتبون الكتاب</code> → <code>كتب كتب</code></td>
</tr>
</tbody>
</table>

### Embedding Options

| Method | Dimensions | Speed | Best For | Command Flag |
|--------|-----------|-------|----------|--------------|
| **TF-IDF** | 5,000 | ⚡ Fast | Large datasets, keyword-focused tasks | `--embed tfidf` |
| **Model2Vec** | 128 | 🐢 Moderate | Semantic understanding, small-medium data | `--embed model2vec` |

---

## 📈 Sample Results

### Training Report Snapshot

```markdown
# 📊 Training Report (Model2Vec ARBERTv2)

## Dataset Information
- **Rows:** 40,046  
- **Classes:** 3  
- **Embedding:** Model2Vec (128 dimensions)

## 🏆 Best Model: Random Forest

**Accuracy:** 78.61%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.75 | 0.71 | 0.73 | 2,840 |
| Neutral | 0.24 | 0.01 | 0.02 | 385 |
| Positive | 0.80 | 0.89 | 0.85 | 4,785 |

**Confusion Matrix:**
```
[[2013    4  823]
 [ 160    4  221]
 [ 496    9 4280]]
```
```

---

## 📁 Output Structure

After running the pipeline, you'll find:

```
outputs/
├── visualizations/
│   ├── eda_class_distribution_pie.png
│   ├── eda_text_length_words.png
│   ├── eda_text_length_chars.png
│   └── eda_top_words.png
├── embeddings/
│   ├── model2vec_embeddings.npy
│   └── model2vec_model.pkl
├── models/
│   └── best_model_model2vec.pkl
└── reports/
    └── training_report_2026-01-17_14-32-08.md
```
---
<div align="center">

**Made with ❤️ for the Arabic NLP
 Week At SDAIA BootCamp**


</div>
