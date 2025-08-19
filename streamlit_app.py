import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Dashboard Validazione Trading", layout="wide")

st.title("📊 Dashboard Validazione Trading")

st.markdown(
    """
    Questa app mostra il test **CSCV / PBO** (Probability of Backtest Overfitting) 
    di López de Prado per valutare se una strategia è overfittata.
    """
)

# -------------------------------
# Utility functions
# -------------------------------
def generate_demo_data(n_strategies=5, n_periods=200, seed=42):
    """Genera un dataset demo di strategie con rendimenti casuali."""
    np.random.seed(seed)
    data = pd.DataFrame(
        {
            f"strategy_{i}": np.random.normal(0.001 * i, 0.02, n_periods)
            for i in range(1, n_strategies + 1)
        }
    )
    data.index = pd.date_range("2020-01-01", periods=n_periods, freq="D")
    return data

def cscv_pbo(data, n_partitions=8):
    """
    Implementazione semplificata del CSCV test.
    Input: dataframe con colonne = strategie, righe = periodi
    """
    n_strat = data.shape[1]
    partitions = np.array_split(data, n_partitions)
    log_ratios = []

    for i in range(n_partitions):
        oos = partitions[i]
        is_ = pd.concat([p for j, p in enumerate(partitions) if j != i])

        # performance medie
        is_perf = is_.mean()
        oos_perf = oos.mean()

        # best IS strategy
        best_strat = is_perf.idxmax()

        # rank of that strategy OOS
        ranks = oos_perf.rank()
        rank_best = ranks[best_strat]

        log_ratio = np.log(rank_best / (n_strat - rank_best + 1))
        log_ratios.append(log_ratio)

    log_ratios = np.array(log_ratios)
    pbo = (log_ratios < 0).mean()
    return log_ratios, pbo

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Carica un file CSV (colonne = strategie, righe = rendimenti)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    st.success("✅ File caricato con successo")
else:
    st.warning("⚠️ Nessun file caricato: uso dataset demo")
    data = generate_demo_data()

# -------------------------------
# Run CSCV / PBO test
# -------------------------------
st.subheader("Risultati CSCV / PBO")

log_ratios, pbo = cscv_pbo(data)

st.write(f"**PBO (Probabilità di Overfitting):** {pbo:.2%}")

# Distribuzione log-ratios
chart_data = pd.DataFrame({"log_ratio": log_ratios})

hist = (
    alt.Chart(chart_data)
    .mark_bar()
    .encode(
        alt.X("log_ratio", bin=alt.Bin(maxbins=20), title="Log(Rank ratio)"),
        y="count()"
    )
    .properties(title="Distribuzione log-ratios (CSCV)", width=600, height=400)
)

st.altair_chart(hist, use_container_width=True)

# -------------------------------
# Preview strategies
# -------------------------------
st.subheader("Anteprima strategie")
st.line_chart((1 + data).cumprod())
