"""
predict.py
Inference engine: loads trained models and generates predictions + auto-responses.
"""

import joblib
import pandas as pd
import numpy as np
import uuid
from preprocess import TicketPreprocessor
from response_templates import get_response


class TicketAnalyzer:
    def __init__(self, model_dir="models"):
        self.preprocessor: TicketPreprocessor = TicketPreprocessor.load(model_dir)
        self.cat_model = joblib.load(f"{model_dir}/category_model.pkl")
        self.sent_model = joblib.load(f"{model_dir}/sentiment_model.pkl")
        self.priority_model = joblib.load(f"{model_dir}/priority_model.pkl")

    def predict(self, subject: str, ticket_text: str, customer_age_days: int = 365,
                previous_tickets: int = 0, channel: str = "Web Form") -> dict:

        df = pd.DataFrame([{
            "subject": subject,
            "ticket_text": ticket_text,
            "customer_age_days": customer_age_days,
            "previous_tickets": previous_tickets,
            "channel": channel,
        }])

        X, df_feat = self.preprocessor.transform(df)

        # Predictions
        cat_pred = self.cat_model.predict(X)[0]
        sent_pred = self.sent_model.predict(X)[0]
        priority_pred = self.priority_model.predict(X)[0]

        # Confidence scores
        cat_proba = self.cat_model.predict_proba(X)[0]
        sent_proba = self.sent_model.predict_proba(X)[0]

        category = self.preprocessor.label_enc_cat.inverse_transform([cat_pred])[0]
        sentiment = self.preprocessor.label_enc_sent.inverse_transform([sent_pred])[0]
        priority = self.preprocessor.label_enc_priority.inverse_transform([priority_pred])[0]

        cat_confidence = round(float(np.max(cat_proba)) * 100, 1)
        sent_confidence = round(float(np.max(sent_proba)) * 100, 1)

        # Category probabilities breakdown
        cat_classes = self.preprocessor.label_enc_cat.classes_
        cat_breakdown = {
            cls: round(float(prob) * 100, 1)
            for cls, prob in zip(cat_classes, cat_proba)
        }

        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
        auto_response = get_response(category, priority, ticket_id)

        return {
            "ticket_id": ticket_id,
            "category": category,
            "category_confidence": cat_confidence,
            "category_breakdown": cat_breakdown,
            "sentiment": sentiment,
            "sentiment_confidence": sent_confidence,
            "priority": priority,
            "urgency_score": int(df_feat["urgency_score"].iloc[0]),
            "auto_response": auto_response,
        }


# ── CLI usage ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = TicketAnalyzer()

    test_cases = [
        {
            "subject": "Charged twice this month",
            "ticket_text": "I was charged twice for my subscription this month. This is extremely urgent and needs immediate attention!",
        },
        {
            "subject": "App keeps crashing",
            "ticket_text": "The mobile app crashes every time I open it after the latest update. Please fix this.",
        },
        {
            "subject": "Order not delivered",
            "ticket_text": "My order was supposed to arrive 3 days ago but I still haven't received it. The tracking shows delivered but it wasn't.",
        },
    ]

    for tc in test_cases:
        result = analyzer.predict(tc["subject"], tc["ticket_text"])
        print(f"\n{'='*60}")
        print(f"Ticket ID  : {result['ticket_id']}")
        print(f"Category   : {result['category']} ({result['category_confidence']}%)")
        print(f"Sentiment  : {result['sentiment']} ({result['sentiment_confidence']}%)")
        print(f"Priority   : {result['priority']}")
        print(f"\nAuto Response:\n{result['auto_response']}")
