# streamlit_app.py — Walk-Forward Bundle (OOS concatenati, step=OOS, cumsum)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import product
from typing import List, Tuple, Dict

st.set_page_config(page_title="Walk-Forward Bundle", layout="wide")
st.title("🚶‍♂️📦 Walk-Forward Bundle — Concatenazione OOS per configurazione")

st.markdown("""
Questa app esegue un **Walk-Forward Bundle (WFB)**:
- Generiamo **più configurazioni** (IS anni × OOS giorni × modalità).
- Per **ogni configurazione**: per ogni split selezioniamo la strategia migliore **in-sample** (secondo una metrica) e **concateniamo i soli rendimenti OOS** del vincitore in **un'unica serie**.
- Calcoliamo le metriche **sulla serie OOS concatenata** (Sharpe, Sortino, Mean, Max Drawdown, CAGR, Hit-rate).
- **Step = OOS** (OOS non sovrapposti per definizione).
""")

# =========================================
# Sidebar — Dati & Griglia configurazioni
# =========================================
with st.sidebar:
    st.header("📦 Dati")
    n_strategies = st.number_input("N° strategie (demo)", 1, 500, 12, 1)
    n_periods    = st.number_input("N° periodi (demo)", 200, 50000, 6000, 100)
    sigma_demo   = st.number_input("Vol demo σ", 0.0001, 0.1, 0.01, 0.0001, format="%.4f")
    seed_demo    = st.number_input("Seed demo", 0, 100000, 42, 1)
    uploaded     = st.file_uploader("📂 CSV (righe=periodi, colonne=strategie)", type=["csv"])
    st.caption("Se non carichi nulla, userò un dataset demo i.i.d. N(0, σ²) dal 2010 in poi.")

    st.divider()
    st.header("🧭 Griglia configurazioni (bundle)")
    start_date   = st.date_input("Data di inizio WFB", value=pd.to_datetime("2013-01-01"))
    is_years_txt = st.text_input("Lista IS (anni)", value="3,5,8")       # es: "3,5,8"
    oos_days_txt = st.text_input("Lista OOS (giorni)", value="63,126")   # es: "63,126"
    modes_sel    = st.multiselect("Modalità", ["sliding","expanding"], default=["sliding"])
    purge_days   = st.number_input("Embargo/Purge (giorni)", 0, 365, 5, 1)
    max_configs  = st.number_input("Limite configurazioni", 1, 500, 50, 1)
    topN_plot    = st.number_input("Top N da plottare", 1, 20, 5, 1)

    st.divider()
    st.header("📏 Metrica di selezione/valutazione")
    metric = st.selectbox("Metrica", ["Sharpe", "Mean return", "Sortino", "Max Drawdown"])
    ann    = st.number_input("Fattore annualizzazione", 1, 252, 252, 1)

# =========================================
# Utils — Dati & Metriche
# =========================================
def generate_demo(n_strats, n_periods, sigma, seed):
    rng = np.random.default_rng(int(seed))
    data = {f"strategy_{i}": rng.normal(0.0, float(sigma), size=int(n_periods)) for i in range(int(n_strats))}
    idx = pd.date_range("2010-01-01", periods=int(n_periods), freq="D")
    return pd.DataFrame(data, index=idx)

def sharpe(mean, std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(std > 0, (mean / std) * np.sqrt(ann), -np.inf)
    return s

def sortino(mean, dn_std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(dn_std > 0, (mean / dn_std) * np.sqrt(ann), -np.inf)
    return s

def max_drawdown_series(ret: pd.Series) -> float:
    if ret.dropna().empty:
        return np.nan
    eq = (1.0 + ret.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (peak - eq) / peak
    return float(dd.max())

def seg_stats(X: np.ndarray, cs: np.ndarray, css: np.ndarray, a: int, b: int):
    """Mean, std, downside-std su tutte le strategie nel segmento [a,b)."""
    n = b - a
    if n <= 1:
        N = X.shape[1]
        return (np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan))
    S  = cs[b]  - cs[a]
    SS = css[b] - css[a]
    mean = S / n
    var  = (SS - (S*S)/n) / (n - 1)
    var  = np.maximum(var, 0.0)
    std  = np.sqrt(var)
    seg = X[a:b]
    dn_std = np.where(seg < 0.0, seg, 0.0).std(axis=0, ddof=1)
    return mean, std, dn_std

def metric_scores(metric, mean, std, dn_std, ann, mdd_vec=None):
    if metric == "Sharpe":       return sharpe(mean, std, ann)
    if metric == "Mean return":  return mean
    if metric == "Sortino":      return sortino(mean, dn_std, ann)
    if metric == "Max Drawdown": return -mdd_vec  # minimizzare MDD ⇒ max(-MDD)
    raise ValueError("Metric not supported")

# =========================================
# Splits (IS/OOS) — step = OOS per definizione (non sovrapposti)
# =========================================
def build_wf_splits(index: pd.DatetimeIndex, start_date: pd.Timestamp,
                    is_years: float, oos_days: int,
                    mode: str, purge_days: int) -> List[Tuple[int,int,int,int]]:
    """
    Ritorna lista di (is_start, is_end, oos_start, oos_end) come POSIZIONI.
    step = oos_days (OOS NON sovrapposti).
    """
    idx = pd.DatetimeIndex(index).sort_values()
    T = len(idx)
    if T == 0: return []

    # prima data utile: start_date ma con almeno is_years anni alle spalle
    min_is_end = idx[0] + pd.DateOffset(years=float(is_years))
    first_oos  = max(pd.to_datetime(start_date), min_is_end)
    oos_start  = idx.searchsorted(first_oos, side="left")
    if oos_start >= T: return []

    splits = []
    while True:
        oos_s = oos_start + int(purge_days)
        oos_e = oos_s + int(oos_days)
        if oos_e > T: break

        if mode == "sliding":
            is_start_date = idx[oos_s] - pd.DateOffset(years=float(is_years))
            is_s = idx.searchsorted(is_start_date, side="left")
            is_e = oos_s
        elif mode == "expanding":
            is_s = 0
            is_e = oos_s
            # richiedi almeno is_years anni effettivi
            if idx[is_e - 1] < (idx[0] + pd.DateOffset(years=float(is_years))):
                oos_start += int(oos_days)  # step = oos_days
                if oos_start >= T: break
                continue
        else:
            raise ValueError("mode must be 'sliding' or 'expanding'")

        if is_e - is_s >= 2:
            splits.append((is_s, is_e, oos_s, oos_e))

        oos_start += int(oos_days)  # step = oos_days (no overlap OOS)
        if oos_start >= T: break

    return splits

# =========================================
# WF per una configurazione → serie OOS concatenata + stats
# =========================================
def run_wf_config_concat_oos(data: pd.DataFrame, start_date, is_years, oos_days,
                             mode, purge_days, metric, ann):
    X = data.to_numpy(copy=False)  # (T, N)
    T, N = X.shape
    cs  = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])
    css = np.vstack([np.zeros((1, N)), np.cumsum(X**2, axis=0)])
    splits = build_wf_splits(data.index, start_date, is_years, oos_days, mode, purge_days)

    # concatenazione dei soli rendimenti OOS del vincitore per ogni split
    oos_concat = pd.Series(index=data.index, dtype=float)  # NaN dove non c'è OOS
    winners = []

    for (i0, i1, o0, o1) in splits:
        mean_is, std_is, dn_is = seg_stats(X, cs, css, i0, i1)

        if metric == "Max Drawdown":
            seg_is = data.iloc[i0:i1]
            mdd_vec = seg_is.apply(lambda s: max_drawdown_series(s), axis=0).to_numpy(dtype=float)
            scores = metric_scores(metric, mean_is, std_is, dn_is, ann, mdd_vec=mdd_vec)
        else:
            scores = metric_scores(metric, mean_is, std_is, dn_is, ann)

        tmp = np.where(np.isfinite(scores), scores, -np.inf)
        w_idx = int(np.nanargmax(tmp))
        winners.append(data.columns[w_idx])

        # append OOS returns of the winner
        oos_concat.iloc[o0:o1] = data.iloc[o0:o1, w_idx].to_numpy()

    # metriche sulla serie OOS concatenata
    oos_ret = oos_concat.dropna()
    stats = {}
    if not oos_ret.empty:
        mu = oos_ret.mean()
        sd = oos_ret.std(ddof=1)
        dn = oos_ret.where(oos_ret < 0, 0.0).std(ddof=1)
        stats["Mean return"]  = float(mu)
        stats["Sharpe"]       = float((mu / sd) * np.sqrt(ann)) if sd > 0 else float("-inf")
        stats["Sortino"]      = float((mu / dn) * np.sqrt(ann)) if dn > 0 else float("-inf")
        stats["Max Drawdown"] = float(max_drawdown_series(oos_ret))
        # CAGR approx sulla finestra coperta dagli OOS
        eq = (1.0 + oos_ret).cumprod()
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        stats["CAGR"] = float(eq.iloc[-1]**(1/years) - 1)
        stats["Hit-rate"] = float((oos_ret > 0).mean())
    else:
        for k in ["Mean return","Sharpe","Sortino","Max Drawdown","CAGR","Hit-rate"]:
            stats[k] = np.nan

    details = {
        "is_years": is_years,
        "oos_days": oos_days,
        "mode": mode,
        "purge_days": purge_days,
        "splits": len(splits),
        "winner_last": winners[-1] if winners else None
    }
    return oos_concat, stats, details

# =========================================
# Carica dati (upload o demo)
# =========================================
if uploaded is not None:
    data = pd.read_csv(uploaded, index_col=0)
    try:
        data.index = pd.to_datetime(data.index)
    except Exception:
        pass
    st.success("✅ File caricato")
else:
    st.info("ℹ️ Nessun file caricato: uso dataset demo i.i.d. N(0, σ²).")
    data = generate_demo(n_strategies, n_periods, sigma_demo, seed_demo)

# =========================================
# Anteprima DEMO — PnL CUMULATO (tutte le strategie)
# =========================================
st.subheader("Anteprima demo — PnL CUMULATO (tutte le strategie)")

cum = data.cumsum()  # cumuliamo i ritorni → PnL che parte da 0
cum_long = (
    cum.reset_index()
       .melt("index", var_name="strategy", value_name="pnl")
       .rename(columns={"index": "date"})
)

chart_cum = (
    alt.Chart(cum_long)
    .mark_line(opacity=0.9)
    .encode(
        x=alt.X("date:T", title="Data"),
        y=alt.Y("pnl:Q", title="PnL cumulato (partenza 0)", scale=alt.Scale(zero=True)),
        color=alt.Color("strategy:N", title="Strategia", legend=None if data.shape[1] > 25 else alt.Legend())
    )
    .properties(height=280)
    .interactive()
)
st.altair_chart(chart_cum, use_container_width=True)


# =========================================
# Costruisci griglia configurazioni (step = OOS)
# =========================================
def parse_list_nums(txt: str):
    vals = []
    for x in txt.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals

IS_list  = parse_list_nums(is_years_txt)               # anni
OOS_list = [int(x) for x in parse_list_nums(oos_days_txt)]  # giorni
modes    = modes_sel if modes_sel else ["sliding"]

grid = list(product(IS_list, OOS_list, modes))  # (IS anni, OOS giorni, mode)
if len(grid) > max_configs:
    grid = grid[:max_configs]
    st.info(f"🔎 Configurazioni limitate a {max_configs} (aumenta 'Limite configurazioni' per elaborarne di più).")

st.write(f"**Configurazioni nel bundle:** {len(grid)} — (step = OOS, OOS non sovrapposti)")

# =========================================
# Esegui il Bundle (cumsum per il plot)
# =========================================
bundle_rows = []
equities: Dict[str, pd.Series] = {}      # config_key -> PnL cumulato OOS concatenato
oos_concat_map: Dict[str, pd.Series] = {}

for (is_y, oos_d, mode) in grid:
    oos_concat, stats, details = run_wf_config_concat_oos(
        data, start_date=pd.to_datetime(start_date),
        is_years=is_y, oos_days=oos_d, mode=mode, purge_days=int(purge_days),
        metric=metric, ann=int(ann)
    )
    key = f"IS={is_y}y | OOS={oos_d}d | {mode}"
    oos_concat_map[key] = oos_concat

    row = {"config": key, **stats, **details}
    bundle_rows.append(row)

    # PnL cumulato della sola serie OOS concatenata (gap = 0 return)
    eq = oos_concat.fillna(0.0).cumsum()  # <-- PnL cumulato: parte da 0
    equities[key] = eq

df_bundle = pd.DataFrame(bundle_rows).sort_values(by="Sharpe", ascending=False)
st.subheader("📊 Classifica configurazioni (serie OOS concatenata)")
st.dataframe(df_bundle.reset_index(drop=True), use_container_width=True)

# =========================================
# Grafici: TopN PnL cumulato OOS concatenati (parte da 0)
# =========================================
if not df_bundle.empty:
    top_keys = df_bundle["config"].head(int(topN_plot)).tolist()
    eq_df = pd.DataFrame({k: equities[k] for k in top_keys})
    eq_long = (
        eq_df.reset_index()
             .melt("index", var_name="config", value_name="equity")
             .rename(columns={"index": "date"})
    )

    chart = (
        alt.Chart(eq_long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Data"),
            y=alt.Y("equity:Q", title="PnL cumulato (solo OOS concatenati)", scale=alt.Scale(zero=True)),
            color=alt.Color("config:N", title="Configurazione")
        )
        .properties(height=360)
        .interactive()
    )
    st.subheader("📈 PnL cumulato — Top configurazioni (OOS concatenati)")
    st.altair_chart(chart, use_container_width=True)

# =========================================
# Distribuzione metrica OOS concatenata
# =========================================
if not df_bundle.empty:
    metric_col = metric if metric in df_bundle.columns else "Sharpe"
    st.subheader(f"Distribuzione della metrica sulla serie OOS concatenata: {metric_col}")
    ch = (
        alt.Chart(df_bundle[[metric_col]])
        .mark_bar()
        .encode(alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=35)), y="count()")
        .properties(height=300)
    )
    st.altair_chart(ch, use_container_width=True)

# =========================================
# Download risultati / OOS concatenati
# =========================================
st.download_button(
    "⬇️ Scarica tabella configurazioni (CSV)",
    data=df_bundle.to_csv(index=False),
    file_name="wfb_bundle_summary.csv",
    mime="text/csv"
)

if not df_bundle.empty:
    out = pd.DataFrame({k: oos_concat_map[k] for k in df_bundle["config"].head(int(topN_plot))})
    st.download_button(
        "⬇️ Scarica OOS concatenati TopN (CSV)",
        data=out.to_csv(),
        file_name="wfb_oos_concat_topN.csv",
        mime="text/csv"
    )
