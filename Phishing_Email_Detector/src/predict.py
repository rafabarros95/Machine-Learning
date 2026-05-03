import joblib
import numpy as np
import re
from nltk.corpus import stopwords


def clean_text(text, stop_words):
    """clean the text by removing stopwords and punctuation and converting to lowercase"""
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words])


def predict_emails(emails):
    """predict the labels for the given emails"""
    stop_words = set(stopwords.words('english'))

    model = joblib.load("src/models/model.pkl")
    vectorizer = joblib.load("src/models/vectorizer.pkl")

    cleaned = [clean_text(email, stop_words) for email in emails]
    vectors = vectorizer.transform(cleaned)
    preds = model.predict(vectors)
    probs = model.predict_proba(vectors)

    results = []
    for text, pred, prob in zip(emails, preds, probs):
        label = "Phishing" if pred == 1 else "Safe"
        confidence = round(np.max(prob), 2)
        results.append((text, label, confidence))
    return results