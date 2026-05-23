"""
generate_dataset.py
Generates a realistic customer support ticket dataset (8,500 samples)
mimicking the Kaggle Customer Support Ticket Dataset structure.
Run this once before training.
"""

import pandas as pd
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

CATEGORIES = {
    "Billing & Payments": [
        "I was charged twice for my subscription this month.",
        "My invoice shows an incorrect amount. Please correct it.",
        "I need a refund for the duplicate transaction on my account.",
        "Why was my credit card charged without authorization?",
        "I cancelled my plan but still got billed. Please help.",
        "My payment failed but money was deducted from my account.",
        "I need a copy of my last 3 invoices for tax purposes.",
        "The discount code I applied was not reflected in my bill.",
        "I upgraded my plan but was charged the old rate.",
        "Please update my billing address to the new one.",
        "I want to switch from annual to monthly billing.",
        "My bank says the transaction was declined but I see a charge.",
        "I need to dispute a charge that I don't recognize.",
        "Can I get a prorated refund for unused days after cancellation?",
        "My auto-renewal failed and now my account is suspended.",
    ],
    "Technical Support": [
        "The application crashes every time I try to open it.",
        "I cannot log in to my account. It says invalid credentials.",
        "The website is loading very slowly on my browser.",
        "I keep getting a 500 internal server error on checkout.",
        "My data is not syncing across devices.",
        "The mobile app is not working after the latest update.",
        "I am unable to upload files. It shows an error every time.",
        "Two-factor authentication is not sending the OTP to my phone.",
        "The dashboard is showing incorrect data and outdated metrics.",
        "I lost all my saved settings after the system update.",
        "The API integration is returning authentication errors.",
        "Video calls keep disconnecting after a few minutes.",
        "I cannot reset my password. The reset email never arrives.",
        "The search feature is not returning relevant results.",
        "My account got locked after too many failed login attempts.",
    ],
    "Account Management": [
        "I want to update my email address on the account.",
        "Please help me delete my account and all associated data.",
        "I need to transfer my account to a different email.",
        "How do I add a secondary user to my account?",
        "I want to change my username but the option is greyed out.",
        "My account was hacked. Please help me secure it immediately.",
        "I need to merge two accounts under the same email.",
        "Can I change the primary contact number on my profile?",
        "I forgot my security question answer and cannot recover my account.",
        "Please remove the old payment method from my account.",
        "I want to enable or disable email notifications.",
        "How do I download all my data before closing the account?",
        "My profile picture is not updating after I upload a new one.",
        "I need to update my company name in the account settings.",
        "Can I have multiple profiles under one account?",
    ],
    "Shipping & Delivery": [
        "My order has not arrived even though it was supposed to be delivered yesterday.",
        "The tracking number provided is not working on the courier website.",
        "I received the wrong item in my package.",
        "My package was marked as delivered but I never received it.",
        "The delivery address on my order is incorrect. Can it be changed?",
        "My order has been stuck in transit for over a week.",
        "I want to cancel my order before it ships.",
        "The item I received was damaged during shipping.",
        "Can I change the delivery date for my upcoming shipment?",
        "I need an urgent delivery but the standard option is too slow.",
        "My order was split into two shipments but I only received one.",
        "The courier attempted delivery but I was not home. How do I reschedule?",
        "I was charged for express shipping but it arrived late.",
        "My return shipment has not been received by your warehouse.",
        "Can I pick up my order from a local store instead of home delivery?",
    ],
    "Product & Service": [
        "The product I received does not match the description on the website.",
        "I want to know more about the premium plan features.",
        "How do I upgrade from the basic to the professional plan?",
        "The product stopped working after just two weeks of use.",
        "I need a replacement for the defective item I received.",
        "Can I get a demo before purchasing the enterprise plan?",
        "What is the difference between the standard and premium subscription?",
        "I want to request a feature that is currently not available.",
        "The product manual is missing from the package.",
        "I am not satisfied with the product quality and want a return.",
        "How long is the warranty period for this product?",
        "I need technical specifications for the product I purchased.",
        "Can I return a product that was bought during a sale?",
        "The product color I received is different from what I ordered.",
        "I want to know if this product is compatible with my existing setup.",
    ],
}

SENTIMENTS = {
    "urgent": [
        "This is extremely urgent and needs immediate attention!",
        "I need this resolved TODAY. This is unacceptable.",
        "This is a critical issue affecting my business operations.",
        "URGENT: Please respond immediately.",
        "I am very frustrated and need an immediate resolution.",
    ],
    "normal": [
        "Please look into this at your earliest convenience.",
        "I would appreciate a response soon.",
        "Kindly help me resolve this issue.",
        "Looking forward to your assistance.",
        "Please let me know how to proceed.",
    ],
    "low": [
        "No rush, but please help when you get a chance.",
        "Just wanted to bring this to your attention.",
        "Whenever possible, please assist with this.",
        "This is not urgent but I would like it resolved.",
        "Please address this when you have time.",
    ],
}

PRIORITY_MAP = {"urgent": "High", "normal": "Medium", "low": "Low"}

def generate_ticket(ticket_id):
    category = random.choice(list(CATEGORIES.keys()))
    base_text = random.choice(CATEGORIES[category])
    sentiment_key = random.choices(
        ["urgent", "normal", "low"], weights=[0.25, 0.55, 0.20]
    )[0]
    suffix = random.choice(SENTIMENTS[sentiment_key])
    ticket_text = f"{base_text} {suffix}"
    subject = base_text[:60] + ("..." if len(base_text) > 60 else "")
    return {
        "ticket_id": f"TKT-{ticket_id:05d}",
        "subject": subject,
        "ticket_text": ticket_text,
        "category": category,
        "sentiment": sentiment_key,
        "priority": PRIORITY_MAP[sentiment_key],
        "customer_age_days": random.randint(1, 1825),
        "previous_tickets": random.randint(0, 15),
        "channel": random.choice(["Email", "Chat", "Phone", "Web Form"]),
    }

def generate_dataset(n=8500):
    records = [generate_ticket(i + 1) for i in range(n)]
    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/customer_tickets.csv", index=False)
    print(f"Dataset generated: {len(df)} tickets saved to data/customer_tickets.csv")
    print(df["category"].value_counts())
    return df

if __name__ == "__main__":
    generate_dataset()
