from pathlib import Path

from Phishing_Email_Detector.src.data_loader import load_data   # for data loading
from Phishing_Email_Detector.src.preprocess import preprocess_df  # for text preprocessing and cleaning
from sklearn.feature_extraction.text import TfidfVectorizer  # for text vectorization
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression   # baseline model for classification
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]  # ./Phishing_Email_Detector
CSV_PATH = BASE_DIR / "data" / "raw" / "Phishing_Email.csv"

def main():
    df = load_data(CSV_PATH)
    df = preprocess_df(df)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))


if __name__ == "__main__":
    main()