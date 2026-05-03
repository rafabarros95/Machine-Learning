import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]  # ./Phishing_Email_Detector
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_loader import load_data
from preprocess import preprocess_df

CSV_PATH = BASE_DIR / "data" / "raw" / "Phishing_Email.csv"

def test_data_loading():
    df = load_data(CSV_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'text' in df.columns and 'label' in df.columns

def test_preprocess_df():
    df = load_data(CSV_PATH)
    processed_df = preprocess_df(df)  # Call the function
    assert 'clean_text' in processed_df.columns  # Ensure the column is present
    assert not processed_df['clean_text'].isnull().any()  # Ensure no null values
    assert all(isinstance(text, str) for text in processed_df['clean_text'])  # Ensure all texts are strings
    assert df["label"].isin([0, 1]).all()  # Ensure labels are 0 or 1