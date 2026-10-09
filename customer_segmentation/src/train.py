"""Train the KMeans customer segmentation model and save it for the Streamlit app.

Run from the repository root:
    uv run python customer_segmentation/src/train.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.pipeline import (  # noqa: E402
    FEATURES, MODEL_PATH, RAW_DATA_PATH, SEGMENT_ORDER,
    build_customers, clean_data, name_clusters, prepare_features,
)

N_CLUSTERS = 4


def main() -> None:
    print("Loading and cleaning data...")
    transactions = clean_data(pd.read_csv(RAW_DATA_PATH))
    customers = build_customers(transactions)
    print(f"{len(customers):,} customers")

    X = prepare_features(customers)
    scaler = StandardScaler().fit(X)

    print(f"Fitting KMeans with k={N_CLUSTERS}...")
    model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42).fit(scaler.transform(X))

    customers["Cluster"] = model.labels_
    customers["Spend_Ratio"] = (customers["Total_Amount"] / (customers["Account_Balance"] + 1)).clip(upper=1)
    segment_names = name_clusters(customers)
    customers["Segment"] = customers["Cluster"].map(segment_names)

    # segment profiles shown in the app
    profiles = customers.groupby("Segment").agg(
        Customers=("Age", "size"),
        Age=("Age", "median"),
        Account_Balance=("Account_Balance", "median"),
        Total_Amount=("Total_Amount", "median"),
        Spend_Ratio=("Spend_Ratio", "median"),
        Female_Share=("Gender", lambda g: (g == "F").mean()),
    ).reindex(SEGMENT_ORDER)
    profiles["Share"] = profiles["Customers"] / profiles["Customers"].sum()

    # context for gender and location (not used by the model)
    gender_mix = pd.crosstab(customers["Gender"], customers["Segment"], normalize="index")[SEGMENT_ORDER]
    location_mix = pd.crosstab(customers["Location"], customers["Segment"], normalize="index")[SEGMENT_ORDER]
    location_counts = customers["Location"].value_counts()

    # small sample of customers for the scatter plot in the app
    plot_sample = customers.sample(5000, random_state=42)[["Age", "Account_Balance", "Total_Amount", "Segment"]]

    bundle = {
        "scaler": scaler,
        "model": model,
        "features": FEATURES,
        "segment_names": segment_names,
        "profiles": profiles,
        "gender_mix": gender_mix,
        "location_mix": location_mix,
        "location_counts": location_counts,
        "plot_sample": plot_sample,
        "n_customers": len(customers),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    print(profiles.round(2))
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
