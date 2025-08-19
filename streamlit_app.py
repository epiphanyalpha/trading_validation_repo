# app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import combinations

st.set_page_config(page_title="Dashboard Validazione Trading", layout="wide")
st.title("📊 Dashboard Validazione Trading")

st.markdown("""
Questa app esegue il test **CSCV / PBO** (Probability of Backtest Overfitting) di López de Prado.
- Divide la serie temporale in *n* partizioni uguali.
- Genera **tutte** le combinazioni IS/OS con IS = OS = n/2.
- Seleziona la strategia **migliore in-sample**, ne valuta il **rank out-of-sample** e calcola:
\\[
\\omega = \\frac{r}{N+1},\\qquad \\lambda = \\log\\frac{\\omega}{1-\\omega},\\qquad \\text{PBO} = \\Pr[\\lambda < 0].
\\]
""")

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("⚙️ Parametri")
    n_partitions = st.number_input("Numero di partizioni (pari)", min_value=2, max_value=24, value=8, step=2)
    metric = st.selectbox("Metrica performance", ["Sharpe", "Mean return"])
    rank_best_is_1 = st.checkbox("Usa convenzione intuitiva: 1 = best", value=True)
    ann_factor = st.number_input("Fattore annualizzazione (Sharpe)", min_value=1, max_value=252, value=252, step=1)
    seed_demo = st.number_input("Seed dataset demo", min_value=0, value=42, step=1)

    # 👉 new controls
    n_strategies_demo = st.number_input("N° strategie (demo)", min_value=1, max_value=200, value=6, step=1)
    n_periods_demo = st.number_input("N° periodi (demo)", min_value=20, max_value=5000, value=250, step=10)
    force_zero_mean = st.checkbox("Forza media 0 (demo)", value=False)


# -------------------------------
# Utilities
# -------------------------------
def generate_demo_data(n_strategies=6, n_periods=250, seed=42, sigma=0.01, force_zero_mean=False):
    """
    Genera dati demo: serie di rendimenti i.i.d. ~ N(0, sigma^2).
    Se force_zero_mean=True, ogni colonna viene de-meanata (media esattamente 0).
    """
    rng = np.random.default_rng(seed)
    data = {
        f"strategy_{i+1}": rng.normal(0.0, sigma, size=n_periods)
        for i in range(n_strategies)
    }
    df = pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n_periods, freq="D"))
    if force_zero_mean:
        df = df - df.mean(axis=0)
    return df


def sharpe_ratio(x: np.ndarray, ann=252, eps=1e-12):
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if sd < eps:
        return -np.inf  # penalizza serie piatte
    return (mu / sd) * np.sqrt(ann)


def perf_series(df: pd.DataFrame, metric: str, ann: int):
    if metric == "Sharpe":
        return df.apply(lambda col: sharpe_ratio(col.values, ann=ann), axis=0)
    else:
        return df.mean(axis=0)


def cscv_pbo(data: pd.DataFrame, n_partitions: int, metric: str, rank_best_is_1: bool, ann: int = 252):
    assert n_partitions % 2 == 0 and n_partitions >= 2, "n_partitions dev'essere un intero pari ≥ 2"
    N = data.shape[1]  # numero strategie
    # split per tempo
    splits = np.array_split(data.index, n_partitions)
    # tutte le combinazioni IS di taglia n/2
    combs_IS = list(combinations(range(n_partitions), n_partitions // 2))
    # OS è il complementare
    combs_OS = [sorted(set(range(n_partitions)) - set(c)) for c in combs_IS]

    logits = []
    details = []

    for IS_idx, OS_idx in zip(combs_IS, combs_OS):
        IS_rows = np.concatenate([splits[i] for i in IS_idx])
        OS_rows = np.concatenate([splits[i] for i in OS_idx])

        df_is = data.loc[IS_rows]
        df_os = data.loc[OS_rows]

        # punteggi per strategia
        s_is = perf_series(df_is, metric, ann)
        winner = s_is.idxmax()

        s_os = perf_series(df_os, metric, ann)

        # rank OOS (1 = best, N = worst) con pandas.rank(ascending=False)
        r_bestfirst = s_os.rank(ascending=False, method="average")[winner]

        # converti alla convenzione del paper (1 = worst, N = best)
        r_paper = N - r_bestfirst + 1 if rank_best_is_1 else r_bestfirst

        # ω in (0,1) usando N+1 per evitare estremi
        omega = float(r_paper) / float(N + 1)
        omega = min(max(omega, 1e-12), 1 - 1e-12)  # sicurezza numerica
        lam = np.log(omega / (1.0 - omega))

        logits.append(lam)
        details.append({
            "IS_partitions": IS_idx,
            "OS_partitions": OS_idx,
            "winner": winner,
            "r_bestfirst": float(r_bestfirst),
            "r_paper": float(r_paper),
            "omega": float(omega),
            "lambda": float(lam),
        })

    logits = np.array(logits, dtype=float)
    pbo = float((logits < 0).mean())
    return logits, pbo, pd.DataFrame(details)


# -------------------------------
# File uploader
# -------------------------------
uploaded = st.file_uploader("📂 Carica CSV (righe=periodi, colonne=strategie)", type=["csv"])

if uploaded is not None:
    data = pd.read_csv(uploaded, index_col=0)
    try:
        data.index = pd.to_datetime(data.index)
    except Exception:
        pass
    st.success("✅ File caricato")
else:
    st.info("ℹ️ Nessun file caricato: uso dataset demo")
    data = generate_demo_data(
        n_strategies=int(n_strategies_demo),
        n_periods=int(n_periods_demo),
        seed=int(seed_demo),
        force_zero_mean=bool(force_zero_mean)
    )


# -------------------------------
# Run CSCV / PBO
# -------------------------------
st.subheader("Risultati CSCV / PBO")
logits, pbo, df_details = cscv_pbo(
    data, n_partitions=n_partitions, metric=metric, rank_best_is_1=rank_best_is_1, ann=ann_factor
)

cols = st.columns(3)
cols[0].metric("PBO (Prob. overfitting)", f"{pbo:.2%}")
cols[1].metric("N° combinazioni", f"{len(logits)}")
cols[2].metric("λ mediano", f"{np.median(logits):.3f}")

# Distribuzione λ
chart_data = pd.DataFrame({"lambda": logits})
hist = (
    alt.Chart(chart_data)
    .mark_bar()
    .encode(
        alt.X("lambda", bin=alt.Bin(maxbins=30), title="λ = logit(ω)"),
        y=alt.Y("count()", title="Frequenza")
    )
    .properties(title="Distribuzione di λ (CSCV)", width=700, height=400)
)
st.altair_chart(hist, use_container_width=True)

# Tabella dettagli
with st.expander("📄 Dettagli combinazioni IS/OS"):
    st.dataframe(df_details)

# -------------------------------
# Anteprima strategie
# -------------------------------
plot_kind = st.radio(
    "Tipo grafico anteprima",
    options=["Equity (cumprod)", "Cumulative returns (cumsum)"],
    index=0,
    horizontal=True
)

st.subheader("Anteprima strategie")

if plot_kind == "Equity (cumprod)":
    to_plot = (1.0 + data).cumprod()
else:
    to_plot = data.cumsum()

st.line_chart(to_plot)

# -------------------------------
# Quick stats
# -------------------------------
st.caption("📊 Statistiche rapide (dataset demo corrente)")
stats = pd.DataFrame({
    "sample_mean": data.mean(),
    "sample_std": data.std(ddof=1),
    "sharpe(ann)": data.apply(lambda c: sharpe_ratio(c.values, ann=ann_factor))
})
st.dataframe(stats.style.format({
    "sample_mean": "{:.6f}",
    "sample_std": "{:.6f}",
    "sharpe(ann)": "{:.3f}"
}))
