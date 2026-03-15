import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import ta

###########################################
# PAGE SETTINGS
###########################################

st.set_page_config(page_title="AI Trading System", layout="wide")

st.title("📈 AI Elliott Wave Trading System")

###########################################
# FEATURE ENGINEERING
###########################################

def add_features(df):

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    df["Return"] = df["Close"].pct_change()

    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    df["Volatility"] = df["Return"].rolling(20).std()

    return df


###########################################
# EXTREMA DETECTION
###########################################

def detect_extrema(df):

    price = df["Close"].values

    maxima = argrelextrema(price, np.greater, order=5)[0]
    minima = argrelextrema(price, np.less, order=5)[0]

    return maxima, minima


###########################################
# WAVE 3 DETECTION
###########################################

def detect_wave3(df):

    maxima, minima = detect_extrema(df)

    waves = sorted(list(maxima) + list(minima))

    df["Wave3"] = 0

    for i in range(4, len(waves)):

        p1, p2, p3, p4, p5 = waves[i-4:i+1]

        if df["Close"].iloc[p3] > df["Close"].iloc[p1] and \
           df["Close"].iloc[p5] > df["Close"].iloc[p3]:

            df.loc[df.index[p3], "Wave3"] = 1

    return df


###########################################
# DIVERGENCE DETECTION
###########################################

def detect_divergence(df):

    df["BullishDiv"] = 0
    df["BearishDiv"] = 0

    maxima, minima = detect_extrema(df)

    for i in range(1, len(minima)):

        p1 = minima[i-1]
        p2 = minima[i]

        if df["Close"].iloc[p2] < df["Close"].iloc[p1] and \
           df["RSI"].iloc[p2] > df["RSI"].iloc[p1]:

            df.loc[df.index[p2], "BullishDiv"] = 1

    for i in range(1, len(maxima)):

        p1 = maxima[i-1]
        p2 = maxima[i]

        if df["Close"].iloc[p2] > df["Close"].iloc[p1] and \
           df["RSI"].iloc[p2] < df["RSI"].iloc[p1]:

            df.loc[df.index[p2], "BearishDiv"] = 1

    return df


###########################################
# FIBONACCI LEVELS
###########################################

def fibonacci_levels(df):

    high = df["Close"].max()
    low = df["Close"].min()

    diff = high - low

    levels = {
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
        "78.6%": high - diff * 0.786
    }

    return levels


###########################################
# LABEL CREATION
###########################################

def create_labels(df):

    df["Signal"] = 0

    df.loc[(df["BullishDiv"] == 1) | (df["Wave3"] == 1), "Signal"] = 1

    df.loc[df["BearishDiv"] == 1, "Signal"] = -1

    return df


###########################################
# TRAIN MODEL
###########################################

def train_model(df):

    features = [
        "RSI",
        "MACD",
        "MACD_signal",
        "MA20",
        "MA50",
        "Momentum",
        "Volatility",
        "Wave3",
        "BullishDiv",
        "BearishDiv"
    ]

    df = df.dropna()

    X = df[features]
    y = df["Signal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False
    )

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return model, acc


###########################################
# BACKTESTING ENGINE
###########################################

def backtest(df):

    capital = 100000
    position = 0

    trades = []

    for i in range(len(df)):

        signal = df["Prediction"].iloc[i]
        price = df["Close"].iloc[i]

        if signal == 1 and position == 0:

            position = capital / price
            capital = 0
            trades.append(("BUY", price))

        elif signal == -1 and position > 0:

            capital = position * price
            position = 0
            trades.append(("SELL", price))

    final_value = capital + position * df["Close"].iloc[-1]

    return final_value, trades


###########################################
# TOP STOCK SELECTOR
###########################################

def top_stocks(df, model):

    features = [
        "RSI",
        "MACD",
        "MACD_signal",
        "MA20",
        "MA50",
        "Momentum",
        "Volatility",
        "Wave3",
        "BullishDiv",
        "BearishDiv"
    ]

    latest = df.groupby("Ticker").tail(1)

    latest = latest.dropna()

    preds = model.predict(latest[features])

    latest["Prediction"] = preds

    buys = latest[latest["Prediction"] == 1]

    return buys.sort_values("RSI").head(10)


###########################################
# PLOT CHART
###########################################

def plot_chart(df):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Price"
    ))

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)


###########################################
# FILE UPLOAD
###########################################

file = st.file_uploader("Upload Dataset (Date | Ticker | Close)", type="csv")

if file:

    df = pd.read_csv(file)

    df["Date"] = pd.to_datetime(df["Date"])

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    tickers = sorted(df["Ticker"].dropna().unique())

    ticker = st.selectbox("Select Stock", tickers)

    stock = df[df["Ticker"] == ticker].copy()

    stock = stock.sort_values("Date")

    ########################################

    stock = add_features(stock)

    stock = detect_wave3(stock)

    stock = detect_divergence(stock)

    stock = create_labels(stock)

    ########################################

    model, acc = train_model(stock)

    st.success(f"Model Accuracy: {round(acc*100,2)} %")

    ########################################

    features = [
        "RSI",
        "MACD",
        "MACD_signal",
        "MA20",
        "MA50",
        "Momentum",
        "Volatility",
        "Wave3",
        "BullishDiv",
        "BearishDiv"
    ]

    stock["Prediction"] = model.predict(stock[features].fillna(0))

    ########################################

    final_value, trades = backtest(stock)

    st.subheader("Backtest Result")

    st.write("Final Portfolio Value:", round(final_value,2))

    ########################################

    st.subheader("Price Chart")

    plot_chart(stock)

    ########################################

    fib = fibonacci_levels(stock)

    st.subheader("Fibonacci Levels")

    st.write(fib)

    ########################################

    best = top_stocks(df, model)

    st.subheader("Top 10 Stocks to Buy Today")

    st.dataframe(best)
