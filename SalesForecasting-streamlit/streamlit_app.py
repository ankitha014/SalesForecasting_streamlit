import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Sales Dashboard", layout="wide")

# =========================
# CUSTOM CSS (DARK + CARDS)
# =========================
st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}

.card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
    transition: 0.3s;
    text-align: center;
}

.card:hover {
    transform: scale(1.05);
}

.metric {
    font-size: 28px;
    font-weight: bold;
}

.label {
    font-size: 14px;
    color: #9ca3af;
}

.fade-in {
    animation: fadeIn 1.2s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='fade-in'>🚀 AI Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_sample():
    return pd.read_csv("Sample_Superstore.csv")

st.download_button(
    "📥 Download Sample CSV",
    load_sample().to_csv(index=False).encode('utf-8'),
    "sample.csv",
    "text/csv"
)

uploaded_file = st.file_uploader("Upload your dataset")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='cp1252')
    st.success("Custom dataset loaded ✅")
else:
    df = load_sample()
    st.info("Using default dataset 📊")

# =========================
# AUTO DETECT COLUMNS
# =========================
def detect_column(names, columns):
    for n in names:
        for c in columns:
            if n.lower() in c.lower():
                return c
    return None

date_col = detect_column(['date', 'order'], df.columns)
sales_col = detect_column(['sales', 'revenue', 'amount'], df.columns)

if date_col is None or sales_col is None:
    st.error("Could not detect Date or Sales columns")
    st.stop()

df.rename(columns={date_col: 'Order Date', sales_col: 'Sales'}, inplace=True)
df['Order Date'] = pd.to_datetime(df['Order Date'])

# =========================
# FILTERS
# =========================
st.sidebar.header("🔎 Filters")

if 'Region' in df.columns:
    region = st.sidebar.selectbox("Region", ["All"] + list(df['Region'].dropna().unique()))
    if region != "All":
        df = df[df['Region'] == region]

if 'Category' in df.columns:
    category = st.sidebar.selectbox("Category", ["All"] + list(df['Category'].dropna().unique()))
    if category != "All":
        df = df[df['Category'] == category]

# =========================
# GRANULARITY
# =========================
st.sidebar.header("⚙️ Settings")
freq = st.sidebar.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"])

if freq == "Daily":
    df_grouped = df.groupby('Order Date')['Sales'].sum().reset_index()
elif freq == "Weekly":
    df_grouped = df.resample('W', on='Order Date')['Sales'].sum().reset_index()
else:
    df_grouped = df.resample('ME', on='Order Date')['Sales'].sum().reset_index()

df_grouped.rename(columns={'Order Date': 'ds', 'Sales': 'y'}, inplace=True)

# =========================
# KPI CARDS
# =========================
total_sales = df['Sales'].sum()
avg_sales = df['Sales'].mean()
growth = ((df_grouped['y'].iloc[-1] - df_grouped['y'].iloc[0]) / df_grouped['y'].iloc[0]) * 100

c1, c2, c3 = st.columns(3)

c1.markdown(f"<div class='card'><div class='label'>Total Sales</div><div class='metric'>{total_sales:,.0f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><div class='label'>Average Sales</div><div class='metric'>{avg_sales:,.2f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div class='label'>Growth %</div><div class='metric'>{growth:.2f}%</div></div>", unsafe_allow_html=True)

# =========================
# MODEL TRAINING
# =========================
if len(df_grouped) < 3:
    st.error("Not enough data points to train the forecasting models. Please adjust your filters or select a finer Time Granularity.")
    st.stop()

model = Prophet()
model.fit(df_grouped)

# =========================
# FORECAST
# =========================
st.subheader("🔮 Forecast")
periods = st.slider("Forecast Days", 30, 365, 90)

freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}  # or 'MS'/'M' depending on pandas version
prophet_freq = freq_map.get(freq, "D")

future = model.make_future_dataframe(periods=periods, freq=prophet_freq)
forecast = model.predict(future)

# =========================
# EVALUATION
# =========================
split = int(len(df_grouped) * 0.8)
train = df_grouped[:split]
test = df_grouped[split:]

model_eval = Prophet()
model_eval.fit(train)

# Predict exactly on the test dates to avoid date misalignment
forecast_eval = model_eval.predict(test[['ds']])
pred = forecast_eval['yhat'].values

mae = mean_absolute_error(test['y'], pred)
rmse = np.sqrt(mean_squared_error(test['y'], pred))

# WMAPE-based accuracy calculation (more robust for volatile sales data)
total_actual = test['y'].sum()
if total_actual == 0:
    accuracy = 0.0
else:
    wmape = np.sum(np.abs(test['y'] - pred)) / total_actual
    accuracy = max(0.0, (1.0 - wmape) * 100.0)

c4, c5, c6 = st.columns(3)

c4.markdown(f"<div class='card'><div class='label'>MAE</div><div class='metric'>{mae:.2f}</div></div>", unsafe_allow_html=True)
c5.markdown(f"<div class='card'><div class='label'>RMSE</div><div class='metric'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
c6.markdown(f"<div class='card'><div class='label'>Accuracy</div><div class='metric'>{accuracy:.2f}%</div></div>", unsafe_allow_html=True)

# =========================
# MODEL COMPARISON
# =========================
st.subheader("🧠 Model Comparison")

results = {}

# Prophet
results["Prophet"] = rmse

# ARIMA
try:
    arima = ARIMA(train['y'], order=(5,1,0)).fit()
    pred_a = arima.forecast(steps=len(test))
    results["ARIMA"] = np.sqrt(mean_squared_error(test['y'], pred_a))
except:
    st.warning("ARIMA failed")

# XGBoost
try:
    df_ml = df_grouped.copy()
    for lag in range(1, 6):
        df_ml[f'lag_{lag}'] = df_ml['y'].shift(lag)
    df_ml.dropna(inplace=True)

    split_ml = int(len(df_ml) * 0.8)
    train_ml = df_ml[:split_ml]
    test_ml = df_ml[split_ml:]

    X_train = train_ml.drop(['ds', 'y'], axis=1)
    y_train = train_ml['y']
    X_test = test_ml.drop(['ds', 'y'], axis=1)
    y_test = test_ml['y']

    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X_train, y_train)

    pred_x = xgb.predict(X_test)
    results["XGBoost"] = np.sqrt(mean_squared_error(y_test, pred_x))
except:
    st.warning("XGBoost failed")

st.bar_chart(pd.DataFrame.from_dict(results, orient='index', columns=['RMSE']))

best_model = min(results, key=results.get)
st.success(f"🏆 Best Model: {best_model}")

# =========================
# PLOTS
# =========================
st.subheader("📉 Forecast")
st.pyplot(model.plot(forecast))

st.subheader("📊 Components")
st.pyplot(model.plot_components(forecast))

# =========================
# DOWNLOAD
# =========================
st.download_button(
    "Download Forecast CSV",
    forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False).encode('utf-8'),
    "forecast.csv",
    "text/csv"
)