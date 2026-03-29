import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Prediction Dashboard", layout="wide")

st.title("📊 Weekly Sales Prediction Dashboard")


# ============================================
# LOAD DATA
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv("weekly_sales.csv")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

@st.cache_data
def load_predictions():
    pred = pd.read_csv("predictions.csv")
    pred['InvoiceDate'] = pd.to_datetime(pred['InvoiceDate'])
    return pred


df = load_data()
pred_df = load_predictions()

st.success("Data Loaded Successfully")


# ============================================
# ACTUAL SALES TREND
# ============================================

st.subheader("📈 Weekly Sales Trend")

fig1, ax1 = plt.subplots()
ax1.plot(df['InvoiceDate'], df['Total_Weekly_Sales'])
ax1.set_title("Actual Sales Over Time")

st.pyplot(fig1)


# ============================================
# MODEL COMPARISON
# ============================================

st.subheader("🤖 Model Prediction Comparison")

model_option = st.selectbox("Select Model", [
    'Linear_Regression_Prediction',
    'Random_Forest_Prediction',
    'XGBoost_Prediction'
])

fig2, ax2 = plt.subplots()

# Actual
ax2.plot(pred_df['InvoiceDate'], pred_df['Actual_Sales'], label="Actual", linewidth=3)

# Selected Model
ax2.plot(pred_df['InvoiceDate'], pred_df[model_option], label=model_option, linestyle='--')

ax2.set_title("Actual vs Predicted Sales")
ax2.legend()

st.pyplot(fig2)


# ============================================
# KPIs
# ============================================

st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{df['Total_Weekly_Sales'].sum():,.0f}")
col2.metric("Average Sales", f"{df['Total_Weekly_Sales'].mean():,.0f}")
col3.metric("Max Sales", f"{df['Total_Weekly_Sales'].max():,.0f}")


# ============================================
# RAW PREDICTIONS TABLE
# ============================================

st.subheader("📂 Prediction Data")

st.dataframe(pred_df.head())


st.write("Developed with Streamlit | ML + Data Warehouse Integration")
