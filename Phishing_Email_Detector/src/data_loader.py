import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]  # .../Phishing_Email_Detector
CSV_PATH = BASE_DIR / "data" / "raw" / "Phishing_Email.csv"


def load_data(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        df = df.dropna(subset=['Email Text'])
        df.columns = ['text', 'label']
        return df
    else:
        raise FileNotFoundError(f"The file at {path} was not found.")


if __name__ == "__main__":
    df = load_data(CSV_PATH)
    # print(df.head())