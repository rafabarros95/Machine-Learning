"""Streamlit app that predicts a bank customer's segment with the trained KMeans model.

Run from the repository root:
    uv run streamlit run customer_segmentation/app.py
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

sys.path.append(str(Path(__file__).resolve().parent))
from src.pipeline import MODEL_PATH, SEGMENT_INFO, SEGMENT_ORDER, predict_segment  # noqa: E402

SEGMENT_COLORS = [SEGMENT_INFO[s]["color"] for s in SEGMENT_ORDER]
GENDERS = {"M": "Male", "F": "Female"}
MIN_CITY_CUSTOMERS = 30  # below this a city's segment mix is too noisy to show

st.set_page_config(page_title="Customer Segmentation", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .hero h1 { margin-bottom: 0.2rem; }
    .hero p { opacity: 0.75; font-size: 1.05rem; margin-top: 0; }
    .segment-card {
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1rem;
        border-left: 8px solid var(--seg-color);
        background: color-mix(in srgb, var(--seg-color) 14%, transparent);
    }
    .segment-card .label {
        text-transform: uppercase; letter-spacing: 0.08em;
        font-size: 0.75rem; opacity: 0.7; margin: 0;
    }
    .segment-card h2 { margin: 0.2rem 0 0.4rem 0; padding: 0; }
    .segment-card h4 { margin: 0 0 0.3rem 0; padding: 0; }
    .segment-card p { margin: 0.2rem 0; }
    .segment-card .share { font-weight: 600; opacity: 0.8; }
    .action {
        border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.6rem;
        background: color-mix(in srgb, var(--seg-color) 8%, transparent);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(MODEL_PATH)


def segment_card(segment: str, title_prefix: str = "", big: bool = False, share: float | None = None) -> None:
    info = SEGMENT_INFO[segment]
    heading = f"<h2>{segment}</h2>" if big else f"<h4>{segment}</h4>"
    label = f'<p class="label">{title_prefix}</p>' if title_prefix else ""
    share_html = f'<p class="share">{share:.0%} of customers</p>' if share is not None else ""
    action = f'<div class="action">💡 {info["action"]}</div>' if big else ""
    st.markdown(
        f"""
        <div class="segment-card" style="--seg-color: {info['color']};">
            {label}{heading}
            <p>{info['description']}</p>
            {share_html}{action}
        </div>
        """,
        unsafe_allow_html=True,
    )


def scatter_chart(bundle: dict, balance: float, total_amount: float) -> alt.LayerChart:
    sample = bundle["plot_sample"]
    sample = sample[(sample["Account_Balance"] > 0) & (sample["Total_Amount"] > 0)]

    log_x = alt.Scale(type="log")
    log_y = alt.Scale(type="log")

    points = alt.Chart(sample).mark_circle(size=22, opacity=0.45).encode(
        x=alt.X("Account_Balance:Q", scale=log_x, title="Account balance (INR, log scale)"),
        y=alt.Y("Total_Amount:Q", scale=log_y, title="Total amount spent (INR, log scale)"),
        color=alt.Color(
            "Segment:N",
            scale=alt.Scale(domain=SEGMENT_ORDER, range=SEGMENT_COLORS),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        tooltip=[
            "Segment",
            "Age",
            alt.Tooltip("Account_Balance:Q", format=",.0f", title="Balance"),
            alt.Tooltip("Total_Amount:Q", format=",.0f", title="Total spent"),
        ],
    )

    # log scales can't show 0, so the customer is drawn at least at 1 INR
    you_df = pd.DataFrame({"Account_Balance": [max(balance, 1)], "Total_Amount": [max(total_amount, 1)], "label": ["You"]})
    you = alt.Chart(you_df).mark_point(shape="diamond", size=450, filled=True, color="black", stroke="white", strokeWidth=2).encode(
        x=alt.X("Account_Balance:Q", scale=log_x), y=alt.Y("Total_Amount:Q", scale=log_y)
    )
    you_label = alt.Chart(you_df).mark_text(dx=18, dy=-14, fontSize=14, fontWeight="bold").encode(
        x=alt.X("Account_Balance:Q", scale=log_x), y=alt.Y("Total_Amount:Q", scale=log_y), text="label"
    )
    return (points + you + you_label).properties(height=440)


def profile_table(bundle: dict, segment: str) -> Styler:
    profiles = bundle["profiles"][["Share", "Age", "Account_Balance", "Total_Amount", "Spend_Ratio", "Female_Share"]]
    profiles = profiles.rename(columns={
        "Share": "Customers",
        "Age": "Median age",
        "Account_Balance": "Median balance (INR)",
        "Total_Amount": "Median total spent (INR)",
        "Spend_Ratio": "Median spend ratio",
        "Female_Share": "Female",
    })

    def highlight(row: pd.Series) -> list[str]:
        if row.name != segment:
            return [""] * len(row)
        return [f"background-color: {SEGMENT_INFO[segment]['color']}55; font-weight: 600"] * len(row)

    return profiles.style.apply(highlight, axis=1).format({
        "Customers": "{:.0%}",
        "Median age": "{:.0f}",
        "Median balance (INR)": "{:,.0f}",
        "Median total spent (INR)": "{:,.0f}",
        "Median spend ratio": "{:.1%}",
        "Female": "{:.0%}",
    })


def mix_chart(mix: pd.DataFrame, title: str) -> alt.Chart:
    """Grouped bars: segment mix of a group (e.g. a city) vs all customers."""
    data = mix.reset_index(names="Segment").melt(id_vars="Segment", var_name="Group", value_name="Share")
    return alt.Chart(data, title=title).mark_bar().encode(
        x=alt.X("Segment:N", sort=SEGMENT_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Group:N"),
        y=alt.Y("Share:Q", axis=alt.Axis(format="%"), title="Share of customers"),
        color=alt.Color("Group:N", scale=alt.Scale(range=["#87CEEB", "#FFA500"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Segment", "Group", alt.Tooltip("Share:Q", format=".1%")],
    ).properties(height=320)


# ---------------------------------------------------------------- layout

st.markdown(
    """
    <div class="hero">
        <h1>🏦 Customer Segmentation</h1>
        <p>Enter a bank customer's details to find out which segment they belong to.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Train it first from the repository root:\n\n"
        "`uv run python customer_segmentation/src/train.py`"
    )
    st.stop()

bundle = load_bundle()
profiles = bundle["profiles"]
cities = bundle["location_counts"].index.tolist()

left, right = st.columns([1, 2.2], gap="large")

with left:
    with st.form("customer_form", border=True):
        st.subheader("Customer details")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        account_balance = st.number_input("Account balance (INR)", min_value=0.0, value=20_000.0, step=1_000.0, format="%.2f")
        total_amount = st.number_input("Total amount spent (INR)", min_value=0.0, value=500.0, step=50.0, format="%.2f")
        gender = st.radio("Gender", options=list(GENDERS), format_func=GENDERS.get, horizontal=True)
        location = st.selectbox("Location", options=cities, format_func=str.title, help="Sorted by number of customers")
        submitted = st.form_submit_button("Predict", type="primary", width="stretch")

    st.caption(
        f"KMeans model with 4 segments, trained on {bundle['n_customers']:,} customers. "
        "The segment is decided by age, balance and spending. Gender and location are shown for context only."
    )

if submitted:
    st.session_state["inputs"] = {
        "age": age,
        "account_balance": account_balance,
        "total_amount": total_amount,
        "gender": gender,
        "location": location,
    }

with right:
    inputs = st.session_state.get("inputs")

    if inputs is None:
        st.subheader("The four customer segments")
        cols = st.columns(2)
        for i, segment in enumerate(SEGMENT_ORDER):
            with cols[i % 2]:
                segment_card(segment, share=profiles.loc[segment, "Share"])
        st.info("👈 Fill in the customer details and press **Predict**.")
    else:
        segment = predict_segment(bundle, inputs["age"], inputs["account_balance"], inputs["total_amount"])
        profile = profiles.loc[segment]
        spend_ratio = min(inputs["total_amount"] / (inputs["account_balance"] + 1), 1)

        segment_card(segment, title_prefix="Predicted segment", big=True, share=profile["Share"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Age", f"{inputs['age']}", f"{inputs['age'] - profile['Age']:+.0f} vs segment median", delta_color="off")
        m2.metric("Balance", f"{inputs['account_balance']:,.0f} INR",
                  f"{inputs['account_balance'] - profile['Account_Balance']:+,.0f} vs median", delta_color="off")
        m3.metric("Total spent", f"{inputs['total_amount']:,.0f} INR",
                  f"{inputs['total_amount'] - profile['Total_Amount']:+,.0f} vs median", delta_color="off")
        m4.metric("Spend ratio", f"{spend_ratio:.1%}",
                  f"{(spend_ratio - profile['Spend_Ratio']) * 100:+.1f} pp vs median", delta_color="off",
                  help="Total spent ÷ account balance")

        tab_chart, tab_profiles, tab_context = st.tabs(["📍 Where the customer fits", "📊 Segment profiles", "👥 Gender & location"])

        with tab_chart:
            st.altair_chart(scatter_chart(bundle, inputs["account_balance"], inputs["total_amount"]), width="stretch")
            st.caption(
                "Each dot is a customer from a 5,000-customer sample. The model also uses age, "
                "so a customer can belong to a segment whose dots are not the closest on this chart."
            )

        with tab_profiles:
            st.dataframe(profile_table(bundle, segment), width="stretch")
            st.caption("Spend ratio = total spent ÷ account balance. The predicted segment is highlighted.")

        with tab_context:
            gender_label = GENDERS[inputs["gender"]]
            gender_share = bundle["gender_mix"].loc[inputs["gender"], segment]
            c1, c2 = st.columns(2)
            c1.metric(f"{gender_label} customers in this segment", f"{gender_share:.0%}",
                      f"{(gender_share - profile['Share']) * 100:+.1f} pp vs all customers", delta_color="off")

            city = inputs["location"]
            city_customers = int(bundle["location_counts"].get(city, 0))
            c2.metric(f"Customers from {city.title()}", f"{city_customers:,}")

            if city_customers >= MIN_CITY_CUSTOMERS:
                mix = pd.DataFrame({
                    city.title(): bundle["location_mix"].loc[city],
                    "All customers": profiles["Share"],
                })
                st.altair_chart(mix_chart(mix, f"Segment mix: {city.title()} vs all customers"), width="stretch")
            else:
                customers_text = "1 customer comes" if city_customers == 1 else f"{city_customers} customers come"
                st.info(f"Only {customers_text} from {city.title()}, too few for a reliable segment breakdown.")

            st.caption("Gender and location are not model features. In the data they barely change which segment a customer is in.")
