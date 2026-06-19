import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Fuel Price AI Dashboard", layout="wide")
st.title("Fuel Price Prediction Dashboard")

st.sidebar.header("Controls")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-12-31"))
lag_days = st.sidebar.slider("Lag Days", 5, 30, 14)
train_ratio = st.sidebar.slider("Train/Test Split", 0.7, 0.95, 0.85)

@st.cache_data
def load_data(start, end):
    df = yf.download("CL=F", start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Close", "Volume"]].dropna()

def create_features(df, lags):
    data = df.copy()
    data["Target_Return"] = np.log(data["Close"].shift(-1) / data["Close"])
    for lag in range(1, lags + 1):
        data[f"return_lag_{lag}"] = data["Close"].pct_change(lag)
    data["Return_1d"] = data["Close"].pct_change()
    data["Volatility_7"] = data["Return_1d"].rolling(7).std()
    data["MA_7"] = data["Close"].rolling(7).mean()
    data["MA_21"] = data["Close"].rolling(21).mean()
    data["Momentum_7"] = data["Close"] - data["Close"].shift(7)
    return data.dropna()

df = load_data(start_date, end_date)

if not df.empty:
    data = create_features(df, lag_days)

    X = data.drop("Target_Return", axis=1)
    y = data["Target_Return"]

    split = int(len(data) * train_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2,
        reg_lambda=3,
        min_child_weight=10,
        random_state=42
    )

    model.fit(X_train, y_train, verbose=False)

    pred_returns = model.predict(X_test)

    actual_prices = df["Close"].shift(-1).loc[X_test.index]
    current_prices = df["Close"].loc[X_test.index]
    predicted_prices = current_prices * np.exp(pred_returns)

    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    st.subheader("Performance")
    c1, c2 = st.columns(2)
    c1.metric("MAE", f"${mae:.2f}")
    c2.metric("R²", f"{r2:.4f}")

    st.subheader("Forecast Accuracy")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=actual_prices.index,
        y=actual_prices,
        mode='lines',
        name='Actual'
    ))

    fig.add_trace(go.Scatter(
        x=predicted_prices.index,
        y=predicted_prices,
        mode='lines',
        name='Predicted'
    ))

    fig.update_layout(
        title="Actual vs Predicted Fuel Price (Crude Oil)",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    latest = X.tail(1)
    next_return = model.predict(latest)[0]

    current_price = df["Close"].iloc[-1]
    tomorrow_prediction = current_price * np.exp(next_return)

    st.subheader("Tomorrow Forecast")

    t1, t2 = st.columns(2)

    t1.metric("Current", f"${current_price:.2f}")

    t2.metric(
        "Tomorrow Prediction",
        f"${tomorrow_prediction:.2f}",
        delta=f"{((tomorrow_prediction-current_price)/current_price)*100:.2f}%"
    )