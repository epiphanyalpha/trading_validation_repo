# app.py — Walk-Forward Bundle Dashboard (no CSCV)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from dataclasses import dataclass
from typing import List, Tuple

st.set_page_config(page_title="Walk-Forward Bundle", layout="wide")
st.title("🚶‍♂️📦 Walk-Forward Bundle — Validazione Strategie")

st.markdown("""
Questa app esegue un **Walk-Forward Bundle (WFB)**:
- L'utente sceglie **data di inizio**, **lunghezza IS minima**, **lunghezza OOS**, **step** (frequenza ricalibrazione), **modalità** (sliding/expanding), **purge/embargo**.
- Per ciascuno split: si seleziona la **migliore strategia in-sample** secondo una **metrica** (Sharpe, return medio, Sortino, Max Drawdown), e si misura la **stessa metrica out-of-sample**.
- Il bundle produce una **distribuzione OOS** (mediana, percentili, hit-rate) e viste interattive.
""")

# =========================================
# Sidebar — Parametri
# =========================================
with st.sidebar:
    st.header("⚙️ Dati demo / Upload")
    n_strategies_demo = st.number_input("N° strategie (demo)", 1, 500, 8, 1)
    n_periods_demo    = st.number_input("N° periodi (demo)", 50, 20000, 1000, 50)
    sigma_demo        = st.number_input("Vol demo σ", 0.0001, 0.1, 0.01, 0.0001, format="%.4f")
    seed_demo         = st.number_input("Seed demo", 0, 10_000, 42, 1)

    uploaded = st.file_uploader("📂 Carica CSV (righe=periodi, colonne=strategie)", type=["csv"])
    st.caption("Se non carichi nulla, userò un dataset demo i.i.d. N(0, σ²).")

    st.divider()
    st.header("🧭 Walk-Forward")
    start_date   = st.date_input("Data di inizio WFB", value=pd.to_datetime("2013-01-01"))
    min_is_years = st.number_input("IS minimo (anni)", 0.5, 50.0, 5.0, 0.5)
    test_len_days= st.number_input("Lunghezza OOS (giorni)", 5, 5000, 63, 1)
    step_days    = st.number_input("Step tra split (giorni)", 1, 5000, 63, 1)
    wf_mode      = st.selectbox("Modalità", ["sliding", "expanding"])
    purge_days   = st.number_input("Embargo/Purge (giorni)", 0, 365, 5, 1)

    insuff_policy= st.selectbox("Policy IS insufficiente", [
        "shift_start (consigliata)", "strict", "shorten_IS"
    ])

    st.divider()
    st.header("📏 Metrica")
    metric = st.selectbox("Metrica di selezione / valutazione", [
        "Sharpe", "Mean return", "Sortino", "Max Drawdown"
    ])
    ann_factor = st.number_input("Fattore annualizzazione (Sharpe/Sortino)", 1, 252, 252, 1)

    st.divider()
    st.header("🏃 Performance")
    max_splits = st.number_input("Limite split (0 = tutti)", 0, 10000, 0, 10)
    cache_toggle = st.checkbox("Cache risultati (veloce)", value=True)

# =========================================
# Utilities — Dati & Metriche
# =========================================
def generate_demo_data(n_strategies=6, n_periods=1000, sigma=0.01, seed=42):
    rng = np.random.default_rng(int(seed))
    data = {
        f"strategy_{i}": rng.normal(0.0, float(sigma), size=int(n_periods))
        for i in range(int(n_strategies))
    }
    idx = pd.date_range("2010-01-01", periods=int(n_periods), freq="D")
    return pd.DataFrame(data, index=idx)

def cum_stats_segment(X, cs, css, a, b):
    """
    Statistiche per il segmento [a, b) su tutte le strategie (colonne di X):
    - mean, std (unbiased)
    - downside std (per Sortino, soglia 0)
    """
    n = b - a
    if n <= 1:
        N = X.shape[1]
        nan = np.nan
        return (np.full(N, nan), np.full(N, nan), np.full(N, nan))
    S  = cs[b]  - cs[a]            # (N,)
    SS = css[b] - css[a]           # (N,)
    mean = S / n
    var  = (SS - (S*S)/n) / (n - 1)
    var  = np.maximum(var, 0.0)
    std  = np.sqrt(var)

    # Downside std (returns < 0): calcolo con maschera locale per accuratezza
    seg = X[a:b]                   # (n, N)
    neg = np.where(seg < 0.0, seg, 0.0)
    dn_std = neg.std(axis=0, ddof=1)
    return mean, std, dn_std

def max_drawdown_segment(X, a, b):
    """
    Max Drawdown per strategia nel segmento [a,b):
    Usa equity = cumprod(1+r); MDD = max peak-to-trough (in %).
    Ritorna array (N,) con valori positivi (es. 0.12 = -12%).
    """
    seg = X[a:b]  # (n, N)
    if seg.shape[0] == 0:
        return np.full(seg.shape[1], np.nan)
    equity = (1.0 + seg).cumprod(axis=0)
    peak = np.maximum.accumulate(equity, axis=0)
    dd = (peak - equity) / peak
    return dd.max(axis=0)

def score_from_metric(metric: str, mean, std, dn_std, ann: int, mdd=None):
    sqrt_ann = np.sqrt(float(ann))
    if metric == "Sharpe":
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(std > 0, (mean / std) * sqrt_ann, -np.inf)
    elif metric == "Mean return":
        score = mean
    elif metric == "Sortino":
        with np.errstate(divide='ignore', invalid='ignore'):
            score = np.where(dn_std > 0, (mean / dn_std) * sqrt_ann, -np.inf)
    elif metric == "Max Drawdown":
        # per selezione IS vogliamo *minimizzare* MDD -> trasformo in "più alto è meglio"
        # score = -MDD
        score = -mdd
    else:
        raise ValueError("Metrica non supportata")
    return score

# =========================================
# Build Walk-Forward Splits (con policy)
# =========================================
def build_wf_splits(index: pd.DatetimeIndex,
                    start_date: pd.Timestamp,
                    min_is_years: float,
                    test_len_days: int,
                    step_days: int,
                    mode: str = "sliding",
                    insuff_policy: str = "shift_start (consigliata)",
                    purge_days: int = 0) -> List[Tuple[int,int,int,int]]:
    idx = pd.DatetimeIndex(index).sort_values()
    T = len(idx)
    if T == 0:
        return []

    min_is_end_date = idx[0] + pd.DateOffset(years=float(min_is_years))
    first_oos_candidate = max(pd.to_datetime(start_date), min_is_end_date)
    pos_first_oos = idx.searchsorted(first_oos_candidate, side="left")
    if pos_first_oos >= T:
        return []

    def _find_is_bounds(oos_start_pos):
        if mode == "sliding":
            is_start_date = idx[oos_start_pos] - pd.DateOffset(years=float(min_is_years))
            is_start_pos = idx.searchsorted(is_start_date, side="left")
            is_end_pos = oos_start_pos
            return (is_start_pos, is_end_pos)
        elif mode == "expanding":
            is_start_pos = 0
            is_end_pos = oos_start_pos
            if idx[is_end_pos - 1] < (idx[0] + pd.DateOffset(years=float(min_is_years))):
                return None
            return (is_start_pos, is_end_pos)
        else:
            raise ValueError("mode deve essere 'sliding' o 'expanding'")

    splits = []
    oos_start_pos = pos_first_oos
    while True:
        oos_start_eff = oos_start_pos + int(purge_days)
        if oos_start_eff >= T:
            break
        oos_end = oos_start_eff + int(test_len_days)
        if oos_end > T:
            break

        bounds = _find_is_bounds(oos_start_eff)
        if bounds is None:
            if insuff_policy.startswith("shift_start"):
                oos_start_pos += int(step_days)
                if oos_start_pos >= T:
                    break
                continue
            elif insuff_policy == "strict":
                break
            elif insuff_policy == "shorten_IS":
                is_start_pos, is_end_pos = 0, oos_start_eff
            else:
                break
        else:
            is_start_pos, is_end_pos = bounds

        # IS almeno 2 osservazioni
        if is_end_pos - is_start_pos < 2:
            oos_start_pos += int(step_days)
            if oos_start_pos >= T:
                break
            continue

        splits.append((is_start_pos, is_end_pos, oos_start_eff, oos_end))
        oos_start_pos += int(step_days)
        if oos_start_pos >= T:
            break

    return splits

# =========================================
# Walk-Forward Bundle core (veloce + caching)
# =========================================
@dataclass
class WFBResult:
    oos_scores: np.ndarray
    details: pd.DataFrame

def _run_wfb_core(data: pd.DataFrame, splits, metric: str, ann: int) -> WFBResult:
    X = data.to_numpy(copy=False)           # (T, N)
    T, N = X.shape
    cs  = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])       # (T+1, N)
    css = np.vstack([np.zeros((1, N)), np.cumsum(X**2, axis=0)])    # (T+1, N)

    oos_scores = []
    rows = []

    for k, (i0, i1, o0, o1) in enumerate(splits):
        # IS stats
        mean_is, std_is, dn_is = cum_stats_segment(X, cs, css, i0, i1)
        mdd_is = max_drawdown_segment(X, i0, i1) if metric == "Max Drawdown" else None
        is_scores = score_from_metric(metric, mean_is, std_is, dn_is, ann, mdd=mdd_is)

        # winner (miglior score IS)
        tmp = np.where(np.isfinite(is_scores), is_scores, -np.inf)
        w_idx = int(np.nanargmax(tmp))
        winner = data.columns[w_idx]

        # OOS stats per il winner
        mean_oos, std_oos, dn_oos = cum_stats_segment(X, cs, css, o0, o1)
        if metric == "Max Drawdown":
            mdd_oos_all = max_drawdown_segment(X, o0, o1)
            oos_score = -mdd_oos_all[w_idx]  # coerente con score = -MDD
        else:
            if metric == "Sharpe":
                oos_score = (mean_oos[w_idx] / std_oos[w_idx]) * np.sqrt(ann) if (std_oos[w_idx] > 0 and np.isfinite(mean_oos[w_idx])) else -np.inf
            elif metric == "Mean return":
                oos_score = mean_oos[w_idx]
            elif metric == "Sortino":
                oos_score = (mean_oos[w_idx] / dn_oos[w_idx]) * np.sqrt(ann) if (dn_oos[w_idx] > 0 and np.isfinite(mean_oos[w_idx])) else -np.inf
            else:
                oos_score = np.nan

        oos_scores.append(float(oos_score))
        rows.append({
            "split": k,
            "IS_idx": (i0, i1),
            "OOS_idx": (o0, o1),
            "winner": winner,
            "IS_score_winner": float(is_scores[w_idx]),
            "OOS_score_winner": float(oos_score),
        })

    return WFBResult(oos_scores=np.asarray(oos_scores, dtype=float),
                     details=pd.DataFrame(rows))

@st.cache_data(show_spinner=False)
def run_wfb_cached(data: pd.DataFrame, splits, metric: str, ann: int) -> WFBResult:
    # per caching: indicizza con stringhe
    key_df = data.copy()
    key_df.index = key_df.index.astype(str)
    return _run_wfb_core(key_df, splits, metric, ann)

def run_wfb(data: pd.DataFrame, splits, metric: str, ann: int, use_cache=True) -> WFBResult:
    return run_wfb_cached(data, splits, metric, ann) if use_cache else _run_wfb_core(data, splits, metric, ann)

# =========================================
# Load dati (upload o demo)
# =========================================
if uploaded is not None:
    data = pd.read_csv(uploaded, index_col=0)
    try:
        data.index = pd.to_datetime(data.index)
    except Exception:
        pass
    st.success("✅ File caricato")
else:
    st.info("ℹ️ Nessun file caricato: uso dataset demo i.i.d. N(0, σ²)")
    data = generate_demo_data(n_strategies=n_strategies_demo,
                              n_periods=n_periods_demo,
                              sigma=sigma_demo,
                              seed=seed_demo)

# =========================================
# Build splits & limit numero
# =========================================
splits = build_wf_splits(
    data.index,
    start_date=pd.to_datetime(start_date),
    min_is_years=min_is_years,
    test_len_days=test_len_days,
    step_days=step_days,
    mode=wf_mode,
    insuff_policy=insuff_policy,
    purge_days=purge_days
)
if len(splits) == 0:
    st.warning("Nessuno split generato: aumenta la serie, riduci IS minimo o cambia la policy.")
else:
    if max_splits and len(splits) > max_splits:
        splits = splits[:max_splits]
        st.info(f"🔎 Split limitati a {max_splits} (di {len(splits)} disponibili).")

# =========================================
# Intro “animata” (step-by-step) con Skip
# =========================================
with st.expander("🎬 Intro passo-passo (clicca per aprire)"):
    if len(splits) <= 0:
        st.info("Nessuno split disponibile con i parametri correnti: regola start date / IS minimo / OOS / step.")
    elif len(splits) == 1:
        st.caption("È stato generato un solo split; lo slider non è necessario.")
        demo_idx = 0
    else:
        demo_idx = st.slider("Seleziona split", min_value=0, max_value=len(splits)-1, value=0, step=1)

    if len(splits) > 0:
        # (Per alleggerire, puoi anche sottocampionare righe/strategie qui)
        i0, i1, o0, o1 = splits[demo_idx]
        eq = (1.0 + data).cumprod()
        df_long = (
            eq.reset_index()
              .melt("index", var_name="strategy", value_name="equity")
              .rename(columns={"index":"date"})
        )

        is_start, is_end  = data.index[i0], data.index[i1-1]
        oos_start, oos_end= data.index[o0], data.index[o1-1]
        band = pd.DataFrame({"start":[is_start, oos_start], "end":[is_end, oos_end], "phase":["IS","OOS"]})

        base = alt.Chart(df_long).mark_line().encode(
            x="date:T",
            y=alt.Y("equity:Q", title="Equity (cumprod)"),
            color=alt.Color("strategy:N")
        ).properties(height=300)

        band_chart = alt.Chart(band).mark_rect(opacity=0.12).encode(
            x="start:T", x2="end:T",
            color=alt.Color("phase:N", scale=alt.Scale(range=["#999999","#1f77b4"]))
        )

        st.altair_chart(band_chart + base, use_container_width=True)
        st.caption(f"IS: {is_start.date()} → {is_end.date()}  |  OOS: {oos_start.date()} → {oos_end.date()}")


# =========================================
# Run Walk-Forward Bundle
# =========================================
if len(splits) > 0:
    res = run_wfb(data, splits, metric=metric, ann=ann_factor, use_cache=cache_toggle)

    # KPI
    oos = res.oos_scores
    med = float(np.nanmedian(oos)) if oos.size else np.nan
    p20 = float(np.nanpercentile(oos, 20)) if oos.size else np.nan
    hit = float(np.nanmean(oos > 0.0)) if oos.size else np.nan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Split OOS", f"{len(oos)}")
    c2.metric("Mediana OOS", f"{med:.4f}")
    c3.metric("20° percentile OOS", f"{p20:.4f}")
    c4.metric("Hit-rate (OOS>0)", f"{hit:.0%}")

    # Istogramma OOS
    st.subheader("Distribuzione performance OOS")
    chart_data = pd.DataFrame({"OOS_score": oos})
    hist = (alt.Chart(chart_data)
        .mark_bar()
        .encode(alt.X("OOS_score:Q", bin=alt.Bin(maxbins=40), title=f"Metrica OOS ({metric})"),
                y=alt.Y("count()", title="Frequenza"))
        .properties(height=320))
    st.altair_chart(hist, use_container_width=True)

    # Boxplot per strategia vincitrice (opzionale)
    st.subheader("Vincitori IS per split")
    st.dataframe(res.details[["split","winner","IS_score_winner","OOS_score_winner"]])

    # Download dettagli
    csv = res.details.to_csv(index=False)
    st.download_button("⬇️ Scarica dettagli WFB (CSV)", data=csv, file_name="wfb_details.csv", mime="text/csv")

