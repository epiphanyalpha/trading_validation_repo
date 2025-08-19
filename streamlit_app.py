import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from itertools import product

st.set_page_config(page_title="Dashboard Validazione Trading", layout="wide")

st.title("📊 Dashboard Validazione Trading — PBO Test")

st.markdown(
    """
    Questa app mostra i risultati del **Probability of Backtest Overfitting (PBO)** 
    come descritto da López de Prado.

    - Input atteso: un CSV con `date` come indice e ogni colonna = P&L di una strategia.
    - Se non carichi un file, viene generato un dataset demo (12 strategie sintetiche).
    """
)

# === Load data ===
uploaded_file = st.file_uploader("📂 Carica un file CSV (colonne = strategie)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.success("✅ Dati caricati dal CSV!")
else:
    # DEMO mode: create synthetic data
    np.random.seed(42)
    n_strategies = 12
    n_days = 800
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = pd.DataFrame(
        np.random.randn(n_days, n_strategies).cumsum(axis=0),
        index=dates,
        columns=[f"Strategy_{i}" for i in range(n_strategies)]
    )
    st.info("ℹ️ Nessun file caricato: uso dati demo sintetici.")

st.subheader("📈 Equity Curves (tutte le strategie)")
chart_data = data.reset_index().melt(id_vars="index", var_name="Strategy", value_name="PnL")
chart = (
    alt.Chart(chart_data)
    .mark_line(opacity=0.7)
    .encode(
        x="index:T",
        y="PnL:Q",
        color="Strategy:N",
        tooltip=["index:T", "Strategy", "PnL"]
    )
    .properties(width=800, height=400)
)
st.altair_chart(chart, use_container_width=True)


# === PBO Implementation ===
def pbo_test(df, n_partitions=8):
    """Implements López de Prado's Probability of Backtest Overfitting test."""
    n_strats = df.shape[1]
    n_obs = df.shape[0]

    # Split data into n_partitions
    partition_size = n_obs // n_partitions
    partitions = [df.iloc[i*partition_size:(i+1)*partition_size] for i in range(n_partitions)]

    logit_values = []

    for train_idx in product([0,1], repeat=n_partitions):
        if sum(train_idx) == 0 or sum(train_idx) == n_partitions:
            continue  # skip all-train or all-test

        train_parts = [p for i,p in enumerate(partitions) if train_idx[i]==1]
        test_parts  = [p for i,p in enumerate(partitions) if train_idx[i]==0]

        train = pd.concat(train_parts)
        test  = pd.concat(test_parts)

        # Sharpe ratios
        sr_train = train.mean()/train.std()
        sr_test = test.mean()/test.std()

        # Best strategy in training
        best_idx = sr_train.idxmax()

        # Relative ranking of that strategy in test
        rank_test = sr_test.rank(ascending=True)[best_idx]
        u = (rank_test-1)/(n_strats-1)  # percentile in [0,1]

        if u in [0,1]:
            continue
        logit = np.log(u/(1-u))
        logit_values.append(logit)

    return np.array(logit_values)


# Run PBO
logits = pbo_test(data)

if len(logits) > 0:
    pbo = np.mean(logits <= 0)  # probability that best in-sample is not best out-of-sample
    st.subheader("📊 Risultati PBO")
    st.write(f"**Probability of Backtest Overfitting (PBO): {pbo:.2%}**")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(logits, bins=30, color="skyblue", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", label="logit=0")
    ax.set_title("Distribuzione logit(U)")
    ax.set_xlabel("logit(U)")
    ax.set_ylabel("Frequenza")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("⚠️ Non abbastanza partizioni per calcolare il PBO.")
