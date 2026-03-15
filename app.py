```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.signal import argrelextrema
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor

import ta

st.set_page_config(page_title="AI Trading System", layout="wide")
st.title("📈 Elliott Wave + RSI Divergence ML System")

#################################################
# FEATURE ENGINEERING
#################################################

def add_features(df):

    df = df.copy()

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


#################################################
# EXTREMA
#################################################

def detect_extrema(df):

    price = df["Close"].values

    maxima = argrelextrema(price, np.greater, order=5)[0]
    minima = argrelextrema(price, np.less, order=5)[0]

    return maxima, minima


#################################################
# WAVE 3 DETECTION
#################################################

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


#################################################
# DIVERGENCE
#################################################

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


#################################################
# CREATE LABELS
#################################################

def create_labels(df):

    df["Signal"] = 0

    df.loc[(df["BullishDiv"] == 1) | (df["RSI"] < 30) | (df["Wave3"] == 1), "Signal"] = 1
    df.loc[(df["BearishDiv"] == 1) | (df["RSI"] > 70), "Signal"] = -1

    return df


#################################################
# SIGNAL MODEL (CLASSIFIER)
#################################################

def train_model(df):

    features = [
        "RSI","MACD","MACD_signal","MA20","MA50",
        "Momentum","Volatility","Wave3","BullishDiv","BearishDiv"
    ]

    df = df.dropna().copy()

    X = df[features]
    y = df["Signal"]

    y_encoded = y.replace({-1:0, 0:1, 1:2})

    if len(y_encoded.unique()) < 2:
        raise ValueError("Not enough signal diversity to train model")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, shuffle=False
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective="multi:softmax",
        num_class=3
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    return model, acc


#################################################
# BACKTEST
#################################################

def backtest(df):

    capital = 100000
    position = 0

    for i in range(len(df)):

        signal = df["Prediction"].iloc[i]
        price = df["Close"].iloc[i]

        if signal == 1 and position == 0:
            position = capital / price
            capital = 0

        elif signal == -1 and position > 0:
            capital = position * price
            position = 0

    final_value = capital + position * df["Close"].iloc[-1]

    return final_value


#################################################
# FIBONACCI
#################################################

def fibonacci_levels(df):

    high = df["Close"].max()
    low = df["Close"].min()

    diff = high - low

    levels = {
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50%": high - diff * 0.5,
        "61.8%": high - diff * 0.618
    }

    return levels


#################################################
# CHART
#################################################

def plot_chart(df):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Price"
    ))

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)


#################################################
# TRUE PRICE PREDICTION MODEL
#################################################

def train_price_model(df):

    features = [
        "RSI","MACD","MACD_signal","MA20","MA50",
        "Momentum","Volatility"
    ]

    df = df.dropna().copy()

    df["Target"] = df["Close"].shift(-1)
    df = df.dropna()

    X = df[features]
    y = df["Target"]

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X, y)

    return model


#################################################
# FUTURE PRICE SIMULATION
#################################################

def predict_future_prices(df, model, days=252):

    features = [
        "RSI","MACD","MACD_signal","MA20","MA50",
        "Momentum","Volatility"
    ]

    future_prices = []

    temp_df = df.copy()

    for i in range(days):

        latest = temp_df.iloc[-1:]
        X = latest[features].fillna(0)

        next_price = model.predict(X)[0]

        next_row = latest.copy()

        next_row["Close"] = next_price
        next_row["Date"] = latest["Date"].values[0] + np.timedelta64(1,'D')

        temp_df = pd.concat([temp_df, next_row])

        temp_df = add_features(temp_df)

        future_prices.append(next_price)

    return future_prices


#################################################
# 1 YEAR INVESTMENT ANALYSIS
#################################################

def one_year_return_analysis(df, selected_date, investment=100000):

    df = df.sort_values("Date").reset_index(drop=True)

    df["DateDiff"] = abs(df["Date"] - selected_date)
    start_idx = df["DateDiff"].idxmin()

    start_price = df.loc[start_idx,"Close"]

    end_idx = min(start_idx + 252, len(df)-1)
    end_price = df.loc[end_idx,"Close"]

    actual_value = investment * (end_price / start_price)

    capital = investment
    position = 0

    for i in range(start_idx, end_idx):

        signal = df["Prediction"].iloc[i]
        price = df["Close"].iloc[i]

        if signal == 1 and capital > 0:
            position = capital / price
            capital = 0

        elif signal == -1 and position > 0:
            capital = position * price
            position = 0

    predicted_value = capital + position * df["Close"].iloc[end_idx]

    return start_price,end_price,predicted_value,actual_value,df["Date"].iloc[end_idx]


#################################################
# MAIN
#################################################

file = st.file_uploader("Upload Dataset", type="csv")

if file:

    df = pd.read_csv(file)
    df["Date"] = pd.to_datetime(df["Date"])

    st.dataframe(df.head())

    tickers = sorted(df["Ticker"].dropna().unique())
    ticker = st.selectbox("Select Stock", tickers)

    stock = df[df["Ticker"] == ticker].copy()
    stock = stock.sort_values("Date")

    stock = add_features(stock)
    stock = detect_wave3(stock)
    stock = detect_divergence(stock)
    stock = create_labels(stock)

    try:
        model, acc = train_model(stock)
        st.success(f"Model Accuracy: {round(acc*100,2)} %")

    except Exception as e:
        st.error(str(e))
        st.stop()

    features = [
        "RSI","MACD","MACD_signal","MA20","MA50",
        "Momentum","Volatility","Wave3","BullishDiv","BearishDiv"
    ]

    stock["EncodedPrediction"] = model.predict(stock[features].fillna(0))

    stock["Prediction"] = stock["EncodedPrediction"].replace({
        0:-1,
        1:0,
        2:1
    })

    #################################################
    # BACKTEST
    #################################################

    st.subheader("Backtest")

    final_value = backtest(stock)

    st.write("Final Portfolio Value:", round(final_value,2))

    #################################################
    # PRICE CHART
    #################################################

    st.subheader("Price Chart")

    plot_chart(stock)

    #################################################
    # FIBONACCI
    #################################################

    st.subheader("Fibonacci Levels")

    st.write(fibonacci_levels(stock))

    #################################################
    # 1 YEAR INVESTMENT ANALYSIS
    #################################################

    st.subheader("📅 1 Year Investment Analysis")

    default_date = stock["Date"].iloc[-1].date()

    selected_date = st.date_input(
        "Select Investment Date",
        value=default_date
    )

    investment = st.number_input(
        "Investment Amount",
        value=100000
    )

    if st.button("Run 1 Year Analysis"):

        start_price,end_price,predicted_value,actual_value,end_date = one_year_return_analysis(
            stock,
            pd.to_datetime(selected_date),
            investment
        )

        st.write("Start Price:", round(start_price,2))
        st.write("End Date:", end_date.date())

        col1,col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Portfolio Value",
                round(predicted_value,2)
            )

        with col2:
            st.metric(
                "Actual Portfolio Value",
                round(actual_value,2)
            )

        pred_return = ((predicted_value-investment)/investment)*100
        actual_return = ((actual_value-investment)/investment)*100

        st.write("Predicted Return %:", round(pred_return,2))
        st.write("Actual Return %:", round(actual_return,2))

    #################################################
    # TRUE 1-YEAR PRICE FORECAST
    #################################################

    st.subheader("🔮 AI 1-Year Price Forecast")

    price_model = train_price_model(stock)

    future_prices = predict_future_prices(stock, price_model)

    future_dates = pd.date_range(
        start=stock["Date"].iloc[-1],
        periods=len(future_prices)+1,
        freq="B"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date":future_dates,
        "PredictedPrice":future_prices
    })

    predicted_price_1yr = future_prices[-1]

    current_price = stock["Close"].iloc[-1]

    predicted_return = ((predicted_price_1yr-current_price)/current_price)*100

    st.metric(
        "Predicted Price in 1 Year",
        round(predicted_price_1yr,2)
    )

    st.metric(
        "Expected Return %",
        round(predicted_return,2)
    )

    #################################################
    # FORECAST CHART
    #################################################

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock["Date"],
        y=stock["Close"],
        name="Historical Price"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["PredictedPrice"],
        name="Predicted Price",
        line=dict(dash="dot")
    ))

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)
```
