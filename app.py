import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import ta

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")
st.title("📈 AI Stock Ranking System (1-Year Prediction)")

#############################################
# FEATURE ENGINEERING
#############################################

def add_features(df):

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["Momentum"] = df["Close"] - df["Close"].shift(20)

    df["Volatility"] = df["Close"].pct_change().rolling(20).std()

    return df


#############################################
# CREATE 1 YEAR TARGET
#############################################

def create_target(df):

    # 252 trading days ≈ 1 year

    df["FuturePrice"] = df.groupby("Ticker")["Close"].shift(-252)

    df["FutureReturn"] = (
        df["FuturePrice"] - df["Close"]
    ) / df["Close"]

    return df


#############################################
# TRAIN MODEL
#############################################

def train_model(df):

    features = [
        "RSI",
        "MACD",
        "MACD_signal",
        "MA20",
        "MA50",
        "Momentum",
        "Volatility"
    ]

    df = df.dropna()

    X = df[features]

    y = df["FutureReturn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        shuffle=False,
        test_size=0.2
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    score = r2_score(y_test, pred)

    return model, score


#############################################
# RANK STOCKS
#############################################

def rank_stocks(df, model):

    features = [
        "RSI",
        "MACD",
        "MACD_signal",
        "MA20",
        "MA50",
        "Momentum",
        "Volatility"
    ]

    latest = df.groupby("Ticker").tail(1)

    latest = latest.dropna()

    latest["PredictedReturn"] = model.predict(
        latest[features]
    )

    ranked = latest.sort_values(
        "PredictedReturn",
        ascending=False
    )

    return ranked


#############################################
# PRICE CHART
#############################################

def plot_chart(df):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Price"
        )
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)


#############################################
# STREAMLIT APP
#############################################

file = st.file_uploader("Upload Dataset", type="csv")

if file:

    df = pd.read_csv(file)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker","Date"])

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    #########################################

    df = df.groupby("Ticker").apply(add_features)

    df = create_target(df)

    #########################################

    st.subheader("Training Model on ALL Stocks")

    model, score = train_model(df)

    st.success(f"Model R² Score: {round(score,3)}")

    #########################################

    st.subheader("Ranking Stocks")

    ranked = rank_stocks(df, model)

    #########################################

    st.subheader("Top 10 Stocks to Buy Today")

    top10 = ranked.head(10)[
        ["Ticker","Close","PredictedReturn"]
    ]

    st.dataframe(top10)

    #########################################

    ticker = st.selectbox(
        "Select Stock Chart",
        ranked["Ticker"].unique()
    )

    stock = df[df["Ticker"] == ticker]

    plot_chart(stock)
