"""Shared data cleaning and feature engineering for the customer segmentation model.

Training (src/train.py) and the Streamlit app (app.py) both use these functions,
so a new customer goes through exactly the same steps as the training data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "raw_data" / "bank_transactions.csv"
MODEL_PATH = PROJECT_DIR / "models" / "kmeans_segments.joblib"

FEATURES = ["Age", "Account_Balance", "Total_Amount", "Spend_Ratio"]
LOG_FEATURES = ["Account_Balance", "Total_Amount"]

SEGMENT_INFO = {
    "Regular spenders": {
        "color": "#87CEEB",  # skyblue
        "description": "Late 20s with a mid-sized balance. They spend the most per customer, "
                       "but only a small share of what they have.",
        "action": "Core customers: reward them with cashback and loyalty programmes.",
    },
    "Small spenders": {
        "color": "#FFA500",  # orange
        "description": "The youngest group, with low balances and very small transactions.",
        "action": "Students and early careers: offer digital wallets and savings products.",
    },
    "Balance drainers": {
        "color": "#FA8072",  # salmon
        "description": "Almost nothing left in the account. They spend nearly their whole balance.",
        "action": "Financially stretched: offer overdraft or micro-credit, and monitor the risk.",
    },
    "Established wealthy": {
        "color": "#3CB371",  # mediumseagreen
        "description": "Mid 40s with the highest balances and moderate spending.",
        "action": "Offer investment, insurance and premium products.",
    },
}
SEGMENT_ORDER = list(SEGMENT_INFO)

COLUMNS = ['CustomerID', 'CustomerDOB', 'CustGender', 'CustLocation', 'CustAccountBalance',
           'TransactionDate', 'TransactionTime', 'TransactionAmount (INR)']


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw transactions (same steps as in eda/eda.ipynb)."""
    df = df[COLUMNS].rename(columns={
        'CustomerID': 'Customer_ID',
        'CustomerDOB': 'Customer_Date_of_Birth',
        'CustGender': 'Gender',
        'CustLocation': 'Location',
        'CustAccountBalance': 'Account_Balance',
        'TransactionDate': 'Transaction_Date',
        'TransactionTime': 'Transaction_Time',
        'TransactionAmount (INR)': 'Transaction_Amount_INR',
    })
    df = df.drop_duplicates().dropna()

    # remove placeholder date of birth
    df = df[df['Customer_Date_of_Birth'] != '1/1/1800']

    # dates are day/month/2-digit-year
    for col in ['Customer_Date_of_Birth', 'Transaction_Date']:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%y')

    # 2-digit years like 57 are parsed as 2057 -> shift them back to the 1900s
    dob = df['Customer_Date_of_Birth']
    df['Customer_Date_of_Birth'] = dob.where(dob <= df['Transaction_Date'], dob - pd.DateOffset(years=100))

    # age at the time of the transaction
    df = df.assign(Age=(df['Transaction_Date'] - df['Customer_Date_of_Birth']).dt.days // 365)
    return df[df["Age"].between(0, 100)]


def build_customers(transactions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the transactions to one row per customer."""
    return transactions.groupby("Customer_ID").agg(
        Age=("Age", "last"),
        Gender=("Gender", "first"),
        Location=("Location", "first"),
        Account_Balance=("Account_Balance", "mean"),
        Total_Amount=("Transaction_Amount_INR", "sum"),
        N_Transactions=("Transaction_Amount_INR", "count"),
    )


def prepare_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Build the KMeans feature matrix X (before scaling)."""
    X = customers[["Age", "Account_Balance", "Total_Amount"]].astype(float)
    X["Spend_Ratio"] = (X["Total_Amount"] / (X["Account_Balance"] + 1)).clip(upper=1)
    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES])
    return X[FEATURES]


def name_clusters(customers: pd.DataFrame) -> dict[int, str]:
    """Map the KMeans cluster ids to segment names, based on each cluster's medians.

    KMeans numbers its clusters arbitrarily, so the names are assigned from the
    profiles instead of hard-coding the ids.
    """
    medians = customers.groupby("Cluster")[["Age", "Total_Amount", "Spend_Ratio"]].median()

    drainers = medians["Spend_Ratio"].idxmax()
    remaining = medians.drop(drainers)
    wealthy = remaining["Age"].idxmax()
    remaining = remaining.drop(wealthy)
    small = remaining["Total_Amount"].idxmin()
    regular = remaining.drop(small).index[0]

    return {
        int(regular): "Regular spenders",
        int(small): "Small spenders",
        int(drainers): "Balance drainers",
        int(wealthy): "Established wealthy",
    }


def predict_segment(bundle: dict, age: float, account_balance: float, total_amount: float) -> str:
    """Predict the segment name of one customer with the trained model bundle."""
    customer = pd.DataFrame({
        "Age": [age],
        "Account_Balance": [account_balance],
        "Total_Amount": [total_amount],
    })
    X = prepare_features(customer)
    cluster = bundle["model"].predict(bundle["scaler"].transform(X))[0]
    return bundle["segment_names"][int(cluster)]
