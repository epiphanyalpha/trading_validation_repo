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

    # controlli demo
    n_strategies_demo = st.number_input("N° strategie (demo)", min_value=1, max_value=200, value=6, step=1)
    n_periods_demo = st.number_input("N° periodi (demo)", min_value=5, max_value=5000, value=250, step=5)

    # limite combinazioni per velocità
    max_cases = st.number_input("Limite combinazioni (None = tutte)", min_value=0, max_value=50000, value=2000, step=100)


# -------------------------------
# Utilities
# -------------------------------
def generate_demo_data(n_strategies=6, n_periods=250, seed=42, sigma=0.01):
    """
    Genera dati demo: serie di rendimenti i.i.d. ~ N(0, sigma^2).
    Nessun drift, tutte le strategie hanno media 0.
    """
    rng = np.random.default_rng(seed)
    data = {
        f"strategy_{i}": rng.normal(0.0, sigma, size=n_periods)
        for i in range(n_strategies)
    }
    df = pd.DataFrame(data, index=pd.date_range("2020-01-01", periods=n_periods, freq="D"))
    return df


def _combine_mean_std(n_vec, sum_mat, sumsq_mat, mask):
    """
    Combina statistiche di partizione in media e std per l'unione definita da mask.
    """
    n = n_vec[mask].sum()
    if n <= 1:
        mean = np.full(sum_mat.shape[1], np.nan, dtype=float)
        std  = np.full(sum_mat.shape[1], np.nan, dtype=float)
        return mean, std

    S  = sum_mat[mask].sum(axis=0)
    SS = sumsq_mat[mask].sum(axis=0)

    mean = S / n
    var = (SS - (S * S) / n) / (n - 1)
    var = np.maximum(var, 0.0)  # numerically safe
    std = np.sqrt(var)
    return mean, std


def cscv_pbo(data: pd.DataFrame, n_partitions: int, metric: str, rank_best_is_1: bool,
             ann: int = 252, max_cases: int | None = None):
    """
    Fast CSCV/PBO:
      - Precompute per-partition sums & sums of squares
      - Combine arithmetically for each IS/OS split
      - Optionally cap the number of cases
    """
    assert n_partitions % 2 == 0 and n_partitions >= 2, "n_partitions dev'essere un intero pari ≥ 2"

    X = data.to_numpy(copy=False)     # (T, N)
    T, N = X.shape
    splits = np.array_split(np.arange(T), n_partitions)
    P = len(splits)

    n_vec = np.array([len(idx) for idx in splits], dtype=np.int64)
    sum_mat  = np.stack([X[idx].sum(axis=0)  for idx in splits], axis=0)   # (P,N)
    sumsq_mat= np.stack([(X[idx]**2).sum(axis=0) for idx in splits], axis=0)

    combs_IS = list(combinations(range(P), P // 2))
    combs_OS = [tuple(sorted(set(range(P)) - set(c))) for c in combs_IS]

    if max_cases and len(combs_IS) > max_cases:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(combs_IS), size=max_cases, replace=False)
        combs_IS = [combs_IS[i] for i in sel]
        combs_OS = [combs_OS[i] for i in sel]

    logits = []
    details_rows = []

    sqrt_ann = np.sqrt(float(ann))

    for IS_idx, OS_idx in zip(combs_IS, combs_OS):
        mask_IS = np.zeros(P, dtype=bool); mask_IS[list(IS_idx)] = True
        mask_OS = ~mask_IS

        mean_IS, std_IS = _combine_mean_std(n_vec, sum_mat, sumsq_mat, mask_IS)
        if metric == "Sharpe":
            sharpe_IS = np.where(std_IS > 0, (mean_IS / std_IS) * sqrt_ann, -np.inf)
            scores_IS = sharpe_IS
        else:
            scores_IS = mean_IS

        tmp = np.where(np.isfinite(scores_IS), scores_IS, -np.inf)
        winner_idx = int(np.nanargmax(tmp))
        winner_name = data.columns[winner_idx]

        mean_OS, std_OS = _combine_mean_std(n_vec, sum_mat, sumsq_mat, mask_OS)
        if metric == "Sharpe":
            sharpe_OS = np.where(std_OS > 0, (mean_OS / std_OS) * sqrt_ann, -np.inf)
            scores_OS = sharpe_OS
        else:
            scores_OS = mean_OS

        order_desc = np.argsort(-scores_OS, kind="mergesort")
        pos = int(np.where(order_desc == winner_idx)[0][0])
        r_bestfirst = 1.0 + pos  # 1..N

        r_paper = (N - r_bestfirst + 1.0) if rank_best_is_1 else r_bestfirst

        omega = float(r_paper) / float(N + 1)
        omega = min(max(omega, 1e-12), 1 - 1e-12)
        lam = np.log(omega / (1.0 - omega))

        logits.append(lam)
        details_rows.append({
            "IS_partitions": IS_idx,
            "OS_partitions": OS_idx,
            "winner": winner_name,
            "r_bestfirst": float(r_bestfirst),
            "r_paper": float(r_paper),
            "omega": float(omega),
            "lambda": float(lam),
        })

    logits = np.asarray(logits, dtype=float)
    pbo = float((logits < 0).mean())
    df_details = pd.DataFrame(details_rows)
    return logits, pbo, df_details


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
        seed=int(seed_demo)
    )


# -------------------------------
# Run CSCV / PBO
# -------------------------------
st.subheader("Risultati CSCV / PBO")
logits, pbo, df_details = cscv_pbo(
    data, n_partitions=n_partitions, metric=metric,
    rank_best_is_1=rank_best_is_1, ann=ann_factor,
    max_cases=None if max_cases == 0 else max_cases
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
st.subheader("Anteprima strategie (cumulata dei rendimenti)")
st.line_chart(data.cumsum())
