# 🎫 Customer Ticket Analysis & Auto Response System

A production-grade ML project that automatically classifies customer support tickets and generates professional responses using an ensemble of LightGBM, XGBoost, and Logistic Regression.

---

## 📌 Project Overview

| Feature | Details |
|---------|---------|
| **Task** | Multi-class ticket classification + auto-response generation |
| **Dataset** | 8,500 synthetic customer support tickets (5 categories) |
| **Models** | Voting Ensemble (LightGBM + XGBoost + Logistic Regression) |
| **Accuracy** | ~95%+ on category classification |
| **Interface** | Streamlit web dashboard |

---

## 🗂️ Project Structure

```
CustomerTicketAI/
├── data/
│   └── customer_tickets.csv       # Generated dataset
├── models/
│   ├── preprocessor.pkl           # Fitted TF-IDF + scaler
│   ├── category_model.pkl         # Ensemble classifier
│   ├── sentiment_model.pkl        # LightGBM sentiment model
│   └── priority_model.pkl         # LightGBM priority model
├── reports/
│   ├── model_results.json         # Accuracy & F1 scores
│   └── *_confusion_matrix.png     # Confusion matrix plots
├── app.py                         # Streamlit dashboard
├── train.py                       # Training pipeline
├── predict.py                     # Inference engine
├── preprocess.py                  # NLP preprocessing
├── generate_dataset.py            # Dataset generator
├── response_templates.py          # Auto-response templates
├── setup_and_run.py               # One-click setup
└── requirements.txt
```

---

## 🚀 Quick Start

### Option 1 — One-click setup
```bash
python setup_and_run.py
```

### Option 2 — Step by step
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_dataset.py

# 3. Train models
python train.py

# 4. Launch dashboard
streamlit run app.py

# 5. (Optional) CLI prediction test
python predict.py
```

---

## 🧠 ML Architecture

### Feature Engineering
- **TF-IDF Vectorizer**: 15,000 features, 1–3 n-grams, sublinear TF scaling
- **Engineered Features**: text length, word count, urgency keyword score, exclamation/question count, caps ratio, customer tenure, previous ticket count

### Models
| Task | Model | Notes |
|------|-------|-------|
| Category | Voting Ensemble | LightGBM(w=2) + XGBoost(w=2) + LR(w=1), soft voting |
| Sentiment | LightGBM | 300 estimators, balanced class weights |
| Priority | LightGBM | 300 estimators, balanced class weights |

### Evaluation
- Stratified 80/20 train-test split
- 5-fold cross-validation
- Metrics: Accuracy, Weighted F1, Confusion Matrix

---

## 📊 Categories & Responses

| Category | Priority Levels | Response SLA |
|----------|----------------|--------------|
| Billing & Payments | High / Medium / Low | 2h / 24h / 3d |
| Technical Support | High / Medium / Low | 1h / 8h / 2d |
| Account Management | High / Medium / Low | 1h / 24h / 3d |
| Shipping & Delivery | High / Medium / Low | 2h / 48h / 3d |
| Product & Service | High / Medium / Low | 24h / 48h / 5d |

---

## 🖥️ Dashboard Features

1. **Analyze Ticket** — Real-time classification + confidence breakdown + auto-response generation
2. **Analytics Dashboard** — Dataset visualizations (category distribution, sentiment, channel, priority)
3. **Model Performance** — Accuracy metrics, confusion matrices, architecture details

---

## 📈 Expected Results

```
Category Classification  → Accuracy: ~95%+  F1: ~0.95
Sentiment Classification → Accuracy: ~92%+  F1: ~0.92
Priority Classification  → Accuracy: ~92%+  F1: ~0.92
```

---

## 🛠️ Tech Stack

- Python 3.10+
- scikit-learn, LightGBM, XGBoost
- NLTK (lemmatization, stopwords)
- Streamlit (dashboard)
- Plotly (visualizations)
- Pandas, NumPy, joblib
