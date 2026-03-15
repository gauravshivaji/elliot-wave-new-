import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from xgboost import XGBClassifier

import ta


################################################
# FEATURE ENGINEERING
################################################

def add_features(df):

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

    df["return"] = df["Close"].pct_change()

    df["volatility"] = df["return"].rolling(10).std()

    df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    df = df.dropna()

    return df


################################################
# ZIGZAG PIVOT DETECTION
################################################

def zigzag(df, pct=0.03):

    prices = df["Close"].values

    pivots = [0]

    last_pivot = 0

    for i in range(1,len(prices)):

        change = (prices[i] - prices[last_pivot]) / prices[last_pivot]

        if abs(change) > pct:

            pivots.append(i)

            last_pivot = i

    return df.iloc[pivots]


################################################
# ELLIOTT WAVE DETECTION
################################################

def detect_waves(pivots):

    waves = []

    for i in range(len(pivots)-8):

        waves.append((i,i+8))

    return waves


################################################
# FIBONACCI RETRACEMENT
################################################

def fibonacci_levels(high,low):

    diff = high - low

    levels = {

        "0.236": high - diff*0.236,
        "0.382": high - diff*0.382,
        "0.5": high - diff*0.5,
        "0.618": high - diff*0.618,
        "0.786": high - diff*0.786

    }

    return levels


################################################
# MODEL TRAINING
################################################

def train_models(df):

    features = ["RSI","return","volatility"]

    X = df[features]

    y = df["target"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,shuffle=False
    )

    ################################
    # RANDOM FOREST
    ################################

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    rf.fit(X_train,y_train)

    rf_pred = rf.predict(X_test)

    ################################
    # XGBOOST
    ################################

    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss"
    )

    xgb.fit(X_train,y_train)

    xgb_pred = xgb.predict(X_test)

    ################################
    # RESULTS
    ################################

    results = pd.DataFrame({

        "Model":[
            "Random Forest",
            "XGBoost"
        ],

        "Accuracy":[
            accuracy_score(y_test,rf_pred),
            accuracy_score(y_test,xgb_pred)
        ],

        "Precision":[
            precision_score(y_test,rf_pred),
            precision_score(y_test,xgb_pred)
        ],

        "Recall":[
            recall_score(y_test,rf_pred),
            recall_score(y_test,xgb_pred)
        ]

    })

    best_model_name = results.sort_values(
        "Accuracy",
        ascending=False
    ).iloc[0]["Model"]

    best_model = rf if best_model_name == "Random Forest" else xgb

    return rf,xgb,best_model,results


################################################
# WAVE-3 PROBABILITY
################################################

def wave3_probability(model,df):

    features = ["RSI","return","volatility"]

    latest = df[features].iloc[-1:]

    prob = model.predict_proba(latest)[0][1]

    return prob


################################################
# BUY / SELL SIGNALS
################################################

def signals(pivots):

    if len(pivots) < 6:

        return None

    wave2 = pivots.iloc[2]

    waveB = pivots.iloc[-2]

    return wave2,waveB


################################################
# NIFTY500 WAVE-3 SCREENER
################################################

def wave3_screener(data):

    candidates = []

    for ticker in data["Ticker"].unique():

        df = data[data["Ticker"]==ticker].dropna(subset=["Close"])

        if len(df) < 150:

            continue

        pivots = zigzag(df)

        if len(pivots) > 6:

            candidates.append(ticker)

    return candidates


################################################
# CHART
################################################

def plot_chart(df,pivots,waves):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Price"
    ))

    fig.add_trace(go.Scatter(
        x=pivots["Date"],
        y=pivots["Close"],
        mode="markers",
        name="Pivots"
    ))

    labels = ["1","2","3","4","5","A","B","C"]

    for w in waves:

        points = pivots.iloc[w[0]:w[1]+1]

        fig.add_trace(go.Scatter(
            x=points["Date"],
            y=points["Close"],
            mode="lines+markers+text",
            text=labels,
            textposition="top center",
            name="Elliott Wave"
        ))

    high = df["Close"].max()

    low = df["Close"].min()

    fib = fibonacci_levels(high,low)

    for level,val in fib.items():

        fig.add_hline(
            y=val,
            line_dash="dash",
            annotation_text=f"Fib {level}"
        )

    return fig


################################################
# STREAMLIT APP
################################################

st.title("📈 Advanced Elliott Wave ML Dashboard")

file = st.file_uploader("Upload NIFTY500 Dataset")

if file:

    data = pd.read_csv(file)

    data["Date"] = pd.to_datetime(data["Date"])

    ticker = st.selectbox(
        "Select Stock",
        sorted(data["Ticker"].dropna().unique())
    )

    df = data[data["Ticker"]==ticker]

    df = df.dropna(subset=["Close"])

    df = df.sort_values("Date")

    if len(df) < 150:

        st.warning("Not enough data")

        st.stop()

    df = add_features(df)

    pivots = zigzag(df)

    waves = detect_waves(pivots)

    rf,xgb,best_model,results = train_models(df)

    prob = wave3_probability(best_model,df)

    fig = plot_chart(df,pivots,waves)

    st.subheader("Stock Price with Elliott Waves")

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Wave-3 Breakout Probability")

    st.metric("Probability",f"{prob*100:.2f}%")

    st.subheader("Model Comparison")

    st.dataframe(results)

    best = results.sort_values("Accuracy",ascending=False).iloc[0]

    st.success(
        f"Best Model: {best['Model']} (Accuracy {best['Accuracy']:.2f})"
    )

    sig = signals(pivots)

    if sig:

        st.subheader("Trading Signals")

        st.write("Buy near Wave-2:",sig[0]["Date"])

        st.write("Buy near Wave-B:",sig[1]["Date"])

    if st.button("Run NIFTY500 Wave-3 Screener"):

        result = wave3_screener(data)

        st.subheader("Stocks Possibly Entering Wave-3")

        st.write(result)
