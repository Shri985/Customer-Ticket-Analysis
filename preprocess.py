"""
preprocess.py
NLP preprocessing pipeline: cleaning, tokenization, TF-IDF + feature engineering.
"""

import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib
import os

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english")) - {"not", "no", "never", "urgent", "immediately"}

URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "critical", "emergency",
    "today", "now", "frustrated", "unacceptable", "serious"
]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["full_text"] = df["subject"].fillna("") + " " + df["ticket_text"].fillna("")
    df["cleaned_text"] = df["full_text"].apply(clean_text)
    df["text_length"] = df["ticket_text"].apply(len)
    df["word_count"] = df["ticket_text"].apply(lambda x: len(str(x).split()))
    df["urgency_score"] = df["ticket_text"].apply(
        lambda x: sum(1 for kw in URGENCY_KEYWORDS if kw in str(x).lower())
    )
    df["exclamation_count"] = df["ticket_text"].apply(lambda x: str(x).count("!"))
    df["question_count"] = df["ticket_text"].apply(lambda x: str(x).count("?"))
    df["caps_ratio"] = df["ticket_text"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    return df


class TicketPreprocessor:
    def __init__(self, max_features=15000, ngram_range=(1, 3)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
            analyzer="word",
        )
        self.label_enc_cat = LabelEncoder()
        self.label_enc_sent = LabelEncoder()
        self.label_enc_priority = LabelEncoder()
        self.scaler = StandardScaler()
        self.numeric_cols = [
            "text_length", "word_count", "urgency_score",
            "exclamation_count", "question_count", "caps_ratio",
            "customer_age_days", "previous_tickets",
        ]

    def fit_transform(self, df: pd.DataFrame):
        df = extract_features(df)
        tfidf_matrix = self.tfidf.fit_transform(df["cleaned_text"])
        numeric_matrix = self.scaler.fit_transform(df[self.numeric_cols].fillna(0))
        X = hstack([tfidf_matrix, csr_matrix(numeric_matrix)])
        y_cat = self.label_enc_cat.fit_transform(df["category"])
        y_sent = self.label_enc_sent.fit_transform(df["sentiment"])
        y_priority = self.label_enc_priority.fit_transform(df["priority"])
        return X, y_cat, y_sent, y_priority, df

    def transform(self, df: pd.DataFrame):
        df = extract_features(df)
        tfidf_matrix = self.tfidf.transform(df["cleaned_text"])
        numeric_matrix = self.scaler.transform(df[self.numeric_cols].fillna(0))
        return hstack([tfidf_matrix, csr_matrix(numeric_matrix)]), df

    def save(self, path="models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "preprocessor.pkl"))

    @staticmethod
    def load(path="models"):
        return joblib.load(os.path.join(path, "preprocessor.pkl"))
