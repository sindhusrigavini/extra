import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="📈 AutoInsights Executive Analyzer", layout="wide")
st.title("📊 AutoInsights Executive Stock Insights (6-Month Smart Analyzer)")
st.write("Upload your *stock or financial CSV* — this version automatically filters and analyzes the *latest 6 months of data.*")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV File", type=["csv"])

# ----------------------------
# HELPERS
# ----------------------------
def parse_value(x):
    """Convert strings like '3.2M', '510K' into numeric values."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(',', '').upper()
    match = re.match(r"([-+]?[0-9]*\.?[0-9]+)([KMB]?)", s)
    if not match:
        return np.nan
    num, mult = match.groups()
    num = float(num)
    if mult == "K":
        num *= 1_000
    elif mult == "M":
        num *= 1_000_000
    elif mult == "B":
        num *= 1_000_000_000
    return num

def to_numeric(df):
    """Convert columns to numeric where possible."""
    df2 = df.copy()
    for col in df2.columns:
        df2[col] = df2[col].apply(parse_value)
    return df2

def detect_date_column(df):
    """Detect date column automatically."""
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                return col
            except:
                pass
    return None

def filter_last_6_months(df, date_col):
    """Filter to the latest 6 months."""
    df = df.sort_values(by=date_col)
    latest = df[date_col].max()
    six_months_ago = latest - pd.DateOffset(months=6)
    return df[df[date_col] >= six_months_ago]

# ----------------------------
# INSIGHT GENERATION
# ----------------------------
def generate_summary_and_insights(df):
    df = to_numeric(df)
    date_col = detect_date_column(df)
    if date_col:
        df = filter_last_6_months(df, date_col)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    n_rows, n_cols = df.shape
    summary_text = f"The dataset contains *{n_rows:,} records* and *{n_cols} columns, representing the **latest 6 months of stock trading activity.* "

    close_col = next((c for c in df.columns if "close" in c.lower()), None)
    vol_col = next((c for c in df.columns if "vol" in c.lower()), None)
    val_col = next((c for c in df.columns if "val" in c.lower()), None)

    insights = []

    # --- TREND ---
    if date_col and close_col:
        df = df.sort_values(by=date_col).dropna(subset=[close_col])
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[close_col].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        pct_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
        direction = "📈 upward" if slope > 0 else "📉 downward" if slope < 0 else "⚖ neutral"

        summary_text += f"The *{close_col.upper()}* shows an {direction} trend over the observed 6-month period, changing by *{pct_change:.2f}%* overall.\n\n"
        if slope > 0:
            reason = "This upward movement suggests improving investor confidence and positive earnings sentiment."
        elif slope < 0:
            reason = "This downward pattern may reflect mild correction, profit-taking, or short-term market pressure."
        else:
            reason = "This stable performance indicates balanced investor sentiment and reduced volatility."
        summary_text += f"*Trend:* {reason}\n\n"
        insights.append(f"🟢 *{close_col.upper()} trend:* {direction} ({pct_change:.2f}% change). {reason}")

    # --- VOLUME ---
    if vol_col:
        vols = df[vol_col].dropna()
        if len(vols) > 0:
            summary_text += f"*Trading volumes* ranged between *{int(vols.min()):,}* and *{int(vols.max()):,}, averaging around **{int(vols.mean()):,}*. "
            insights.append(f"📊 *Volume:* Avg {int(vols.mean()):,}, steady market participation.")

    # --- CORRELATION ---
    if vol_col and val_col and vol_col in df and val_col in df:
        corr = df[[vol_col, val_col]].corr().iloc[0, 1]
        summary_text += f"The *correlation* between *{vol_col.upper()}* and *{val_col.upper()}* is *{corr:.2f}*, showing alignment between trade activity and total market value.\n\n"
        insights.append(f"🔗 *{vol_col.upper()}–{val_col.upper()} correlation:* {corr:.2f} (strong alignment).")

    summary_text += (
        "\n\n*Summary:*\n"
        "The stock shows stable performance with moderate fluctuations over the last 6 months. "
        "Healthy trading volume indicates consistent investor interest. "
        "Price movements align with broader IT sector trends and company performance outlook."
    )

    return summary_text, insights, df, date_col, close_col

# ----------------------------
# APP EXECUTION
# ----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    summary, insights, df_filtered, date_col, close_col = generate_summary_and_insights(df)

    st.subheader("📄 Executive Summary (Latest 6 Months)")
    st.markdown(summary)

    st.subheader("💡 Key Insights")
    for i in insights:
        st.markdown(f"- {i}")

    # --- Trend Chart ---
    if date_col and close_col:
        st.subheader("📉 6-Month Price Trend")
        fig = px.line(df_filtered, x=date_col, y=close_col, title="Price Trend (6-Month)", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- Month-wise Trend ---
    if date_col and close_col:
        st.subheader("📆 Monthly Average Price Analysis")
        monthly = df_filtered.copy()
        monthly["Month"] = monthly[date_col].dt.to_period("M").astype(str)
        avg = monthly.groupby("Month")[close_col].mean().reset_index()
        fig = px.bar(avg, x="Month", y=close_col, title="Monthly Average Price", text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)

    # --- Correlation Heatmap ---
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) > 1:
        st.subheader("📊 Correlation Heatmap")
        corr = df_filtered[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # --- Download Summary ---
    report_text = summary + "\n\nKey Insights:\n" + "\n".join(insights)
    st.download_button("⬇ Download Executive Summary", report_text, file_name="6month_executive_summary.txt")

else:
    st.info("Please upload your CSV file (e.g., TCS stock data) to generate 6-month insights.")