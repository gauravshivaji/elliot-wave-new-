import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier

import ta


############################################
# LOAD DATASET
############################################

def load_dataset(file):

    df = pd.read_csv(file)

    # ensure first column is Date
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # convert date safely
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(subset=["Date"])

    # convert wide → long format
    if "Stock" not in df.columns:

        df = df.melt(
            id_vars=["Date"],
            var_name="Stock",
            value_name="Close"
        )

    # ensure numeric prices
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna()

    return df


############################################
# FEATURE ENGINEERING
############################################

def add_features(df):

    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    ).rsi()

    df["Return"] = df["Close"].pct_change()

    df["Fib_Ratio"] = (
        df["Close"] - df["Close"].rolling(30).min()
    ) / (
        df["Close"].rolling(30).max()
        - df["Close"].rolling(30).min()
    )

    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    df = df.dropna()

    return df


############################################
# PIVOT DETECTION
############################################

def detect_pivots(df):

    if len(df) < 50:
        return pd.DataFrame()

    order = 25

    highs = argrelextrema(
        df["Close"].values,
        np.greater,
        order=order
    )[0]

    lows = argrelextrema(
        df["Close"].values,
        np.less,
        order=order
    )[0]

    pivots = sorted(list(highs) + list(lows))

    return df.iloc[pivots]


############################################
# ELLIOTT WAVE DETECTION
############################################

def detect_elliott(pivots):

    cycles = []

    if len(pivots) < 6:
        return cycles

    prices = pivots["Close"].values

    for i in range(len(prices)-5):

        p1,p2,p3,p4,p5,p6 = prices[i:i+6]

        w1 = p2 - p1
        w2 = p3 - p2
        w3 = p4 - p3
        w4 = p5 - p4
        w5 = p6 - p5

        if w1 == 0:
            continue

        r2 = abs(w2/w1)
        r3 = abs(w3/w1)
        r4 = abs(w4/w3) if w3 != 0 else 0
        r5 = abs(w5/w1)

        cond1 = 0.4 < r2 < 0.7
        cond2 = r3 > 1.3
        cond3 = 0.2 < r4 < 0.5
        cond4 = r5 > 0.5

        if cond1 and cond2 and cond3 and cond4:
            cycles.append((i,i+5))

    return cycles


############################################
# TRAIN MODELS
############################################

def train_models(df):

    if len(df) < 100:
        return None

    features = ["RSI","Fib_Ratio","Return"]

    X = df[features]
    y = df["Target"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    ################################
    # Random Forest
    ################################

    rf = RandomForestClassifier(n_estimators=300)
    rf.fit(X_train,y_train)

    rf_pred = rf.predict(X_test)

    ################################
    # XGBoost
    ################################

    xgb = XGBClassifier(n_estimators=400)
    xgb.fit(X_train,y_train)

    xgb_pred = xgb.predict(X_test)

    ################################
    # comparison table
    ################################

    results = pd.DataFrame({

        "Model":["Random Forest","XGBoost"],

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

    return results


############################################
# PLOT
############################################

def plot_chart(df,pivots,cycles):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Price"
        )
    )

    if len(pivots) > 0:

        fig.add_trace(
            go.Scatter(
                x=pivots["Date"],
                y=pivots["Close"],
                mode="markers",
                name="Pivots"
            )
        )

    for c in cycles:

        wave = pivots.iloc[c[0]:c[1]+1]

        fig.add_trace(
            go.Scatter(
                x=wave["Date"],
                y=wave["Close"],
                mode="lines+markers",
                name="Elliott Cycle"
            )
        )

    return fig


############################################
# STREAMLIT APP
############################################

st.title("📈 Elliott Wave + Fibonacci ML Dashboard")

uploaded = st.file_uploader("Upload 6-Year Dataset")

if uploaded:

    df = load_dataset(uploaded)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    stocks = df["Stock"].unique()

    if len(stocks) == 0:
        st.error("No stocks detected in dataset.")
        st.stop()

    stock = st.selectbox("Select Stock", stocks)

    df_stock = df[df["Stock"] == stock]

    df_stock = df_stock.sort_values("Date")

    df_stock = add_features(df_stock)

    if len(df_stock) < 100:
        st.warning("Not enough data for model training.")
        st.stop()

    pivots = detect_pivots(df_stock)

    cycles = detect_elliott(pivots)

    results = train_models(df_stock)

    fig = plot_chart(df_stock,pivots,cycles)

    st.subheader("Stock Price with Elliott Waves")
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Model Comparison")

    if results is not None:

        st.dataframe(results)

        best = results.sort_values(
            "Accuracy",
            ascending=False
        ).iloc[0]

        st.success(
            f"Best Model: {best['Model']} "
            f"(Accuracy {best['Accuracy']:.2f})"
        )
