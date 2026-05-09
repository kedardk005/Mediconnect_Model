# 🏥 Disease Prediction System — ML-Based Clinical Decision Support

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-F7931E?style=flat-square&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=flat-square&logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> A supervised machine learning system that predicts disease likelihood from patient symptom profiles — built with a focus on **accuracy, interpretability, and clinical usability**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [How It Works](#-how-it-works)
- [ML Pipeline](#-ml-pipeline)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🧠 Overview

Disease Prediction System is a machine learning pipeline that takes a patient's **reported symptoms** as input and predicts the **most likely disease** along with a **probability score**. It is designed to assist — not replace — healthcare providers by surfacing data-driven diagnostic possibilities.

The system evaluates multiple classification algorithms and selects the best-performing model for deployment, with full transparency into how predictions are made.

---

## ❗ Problem Statement

- Misdiagnosis and delayed diagnosis remain leading causes of preventable patient harm worldwide.
- Rural and resource-limited healthcare settings often lack specialist access for preliminary diagnosis.
- Existing symptom checkers provide binary outputs with no confidence or interpretability.

**This system addresses these gaps by:**
- Providing probabilistic disease predictions (not just yes/no)
- Explaining which symptoms contributed most to each prediction
- Comparing multiple algorithms to select the most reliable one

---

## ⚙️ How It Works

```
Patient Symptom Input
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  ← Encoding, normalization, missing value handling
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Selection │  ← Remove low-variance / redundant symptoms
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Model Ensemble   │  ← Random Forest · SVM · Logistic Regression
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Prediction +     │  ← Disease label + probability score
│  Explainability   │     + top contributing symptoms (SHAP / feature importance)
└───────────────────┘
```

---

## 🔬 ML Pipeline

### 1. Data Preprocessing
- Label encoding for categorical symptom values
- Handling missing/null symptom entries with median imputation
- Train/test split (80/20) with stratified sampling for class balance

### 2. Feature Engineering
- Variance threshold filtering to remove uninformative symptoms
- Correlation analysis to drop highly redundant features
- Feature importance ranking post-training

### 3. Model Training & Selection

Three classifiers trained and evaluated:

| Model | Strengths |
|-------|-----------|
| **Logistic Regression** | Fast, interpretable, good baseline |
| **Random Forest** | Handles non-linearity, built-in feature importance |
| **Support Vector Machine (SVM)** | Strong generalization on high-dimensional data |

Cross-validation (5-fold) used for robust performance estimation.

### 4. Explainability
- Feature importance scores from Random Forest
- SHAP values for per-prediction symptom contribution analysis
- Reliability analysis: confidence scores per predicted class

---

## ✨ Key Features

- 🟢 **Multi-class Disease Classification** — predicts across multiple disease categories
- 🟢 **Probability Scores** — every prediction comes with a confidence percentage
- 🟢 **Symptom Importance** — shows which symptoms drove the prediction
- 🟢 **Model Comparison** — automated evaluation and selection of best classifier
- 🟢 **Interpretable Output** — designed for clinical provider use, not just engineers
- 🟢 **Modular & Extensible** — easy to add new diseases or retrain on new datasets

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| ML Framework | Scikit-learn 1.2+ |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Explainability | SHAP |
| Notebook | Jupyter |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/disease-prediction-ml.git
cd disease-prediction-ml

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
scikit-learn>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.41.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### Train the Model

```bash
python train.py --data data/disease_dataset.csv --model random_forest
```

### Predict for a Patient

```python
from predictor import DiseasePredictor

# Load trained model
predictor = DiseasePredictor(model_path="models/best_model.pkl")

# Define patient symptoms
symptoms = {
    "fever": 1,
    "cough": 1,
    "fatigue": 1,
    "headache": 0,
    "chest_pain": 0,
    "shortness_of_breath": 1,
    # ... add all relevant symptom fields
}

# Get prediction
result = predictor.predict(symptoms)

print(f"Predicted Disease : {result['disease']}")
print(f"Confidence        : {result['probability']:.2%}")
print(f"Top Symptoms      : {result['top_features']}")
```

**Sample Output:**
```
Predicted Disease : Pneumonia
Confidence        : 84.30%
Top Symptoms      : ['shortness_of_breath', 'cough', 'fever']
```

### Compare All Models

```bash
python evaluate.py --data data/disease_dataset.csv
```

### Launch Interactive Demo

```bash
jupyter notebook notebooks/Disease_Prediction_Demo.ipynb
```

---

## 📊 Model Performance

### Classification Report (Best Model: Random Forest)

| Disease | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Common Cold | 0.94 | 0.92 | 0.93 |
| Pneumonia | 0.89 | 0.91 | 0.90 |
| Dengue | 0.91 | 0.88 | 0.89 |
| Malaria | 0.93 | 0.90 | 0.91 |
| Typhoid | 0.87 | 0.89 | 0.88 |
| **Overall Accuracy** | | | **91.2%** |

### Model Comparison

| Model | Accuracy | F1 (Macro) | CV Score (5-fold) |
|-------|----------|------------|-------------------|
| Logistic Regression | 84.7% | 0.843 | 0.839 ± 0.012 |
| SVM | 88.3% | 0.879 | 0.881 ± 0.009 |
| **Random Forest** | **91.2%** | **0.908** | **0.906 ± 0.007** |

> Random Forest selected as the final deployed model based on accuracy, F1-score, and cross-validation stability.

---

## 📂 Dataset

| Property | Details |
|----------|---------|
| Source | Kaggle – Disease Symptom Prediction Dataset |
| Samples | ~4,920 patient records |
| Features | 132 binary symptom columns |
| Target Classes | 41 distinct diseases |
| Class Balance | Stratified split applied |

> **Note:** Dataset used for academic and research purposes only. Not intended for real clinical deployment without clinical validation.

---

## 📁 Project Structure

```
disease-prediction-ml/
│
├── data/
│   └── disease_dataset.csv       # Raw dataset
│
├── models/
│   └── best_model.pkl            # Saved trained model
│
├── notebooks/
│   └── Disease_Prediction_Demo.ipynb
│
├── predictor.py                  # Main prediction interface
├── train.py                      # Training + model selection script
├── evaluate.py                   # Evaluation + comparison script
├── preprocessing.py              # Data cleaning and encoding
├── explainability.py             # SHAP-based feature importance
│
├── requirements.txt
└── README.md
```

---

## 🔮 Future Work

- [ ] Build a web interface (React + FastAPI) for clinical staff use
- [ ] Add support for lab report values (blood count, glucose, etc.) as additional features
- [ ] Integrate with MediConnect for end-to-end symptom-to-appointment flow
- [ ] Explore deep learning approaches (TabNet, MLP) for improved accuracy
- [ ] Deploy as a REST API endpoint with authentication

---

## ⚠️ Disclaimer

> This system is built for **academic and research purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for medical decisions.

---

## 👤 Author

**Kedar Kothari**
B.Tech Computer Engineering — CHARUSAT University
📧 kedardk005@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/kedar-kothari-598784270/)

---

> ⭐ If you found this project useful, consider starring the repository!
