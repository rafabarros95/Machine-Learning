import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]  # ./Phishing_Email_Detector
CSV_PATH = BASE_DIR / "data" / "raw" / "Phishing_Email.csv"

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))


if __name__ == '__main__':
    from data_loader import load_data
    from preprocess import preprocess_df
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    BASE_DIR = Path(__file__).resolve().parents[1]  # .../Phishing_Email_Detector
    MODELS_DIR = BASE_DIR / "src" / "models"

    df = load_data(CSV_PATH)
    df = preprocess_df(df)

    vectorizer = joblib.load(MODELS_DIR/"vectorizer.pkl")
    X = vectorizer.transform(df['clean_text'])
    y = df['label']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(MODELS_DIR/"model.pkl")
    evaluate_model(model, X_test, y_test)