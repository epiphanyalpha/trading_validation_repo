# streamlit_app.py — Walk-Forward Bundle (v2)
# Major upgrades: embargo fix, vectorized MDD, progress bar, heatmap, better sorting, caching, robust CSV, monthly/annual units

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import product
from typing import List, Tuple, Dict, Optional
import re

# ============================
# Page / Title
# ============================
st.set_page_config(page_title="Walk-Forward Bundle (v2)", layout="wide")
st.title("🚶‍♂️📦 Walk-Forward Bundle — Concatenazione OOS per configurazione (v2)")

st.markdown(
    """
Questa app esegue un **Walk-Forward Bundle (WFB)** con **step = OOS** (OOS non sovrapposti). Per ogni
configurazione: selezioniamo in-sample la strategia migliore secondo una metrica e **concateniamo solo i rendimenti OOS**
del vincitore in **un'unica serie**. Calcoliamo le metriche sulla serie OOS concatenata (Sharpe, Sortino, Mean, Max Drawdown, CAGR, Hit-rate).

**Novità v2**: fix embargo, unità tempo (giorni/mesi/anni), MDD vettoriale, heatmap metrica, caching, caricamento CSV robusto,
barra di avanzamento, ordinamento per metrica selezionata, pannello metriche "Top 1".
"""
)

# ============================
# Sidebar — Dati & Griglia
# ============================
with st.sidebar:
    st.header("📦 Dati")
    n_strategies = st.number_input("N° strategie (demo)", 1, 500, 12, 1)
    n_periods    = st.number_input("N° periodi (demo)", 200, 100_000, 6000, 100)
    sigma_demo   = st.number_input("Vol demo σ", 0.0001, 0.5, 0.01, 0.0001, format="%.4f")
    seed_demo    = st.number_input("Seed demo", 0, 1_000_000, 42, 1)
    uploaded     = st.file_uploader("📂 CSV (righe=periodi, colonne=strategie)", type=["csv"])    
    st.caption("Se non carichi nulla, userò un dataset demo i.i.d. N(0, σ²) dal 2010 in poi.")

    st.divider()
    st.header("🧭 Griglia configurazioni (bundle)")
    start_date   = st.date_input("Data di inizio WFB", value=pd.to_datetime("2013-01-01"))
    time_unit    = st.radio("Unità di misura IS/OOS", ["giorni", "mesi", "anni"], index=0, horizontal=True)
    is_txt       = st.text_input("Lista IS (unità sopra)", value="3,5,8")
    oos_txt      = st.text_input("Lista OOS (unità sopra)", value="63,126")
    modes_sel    = st.multiselect("Modalità", ["sliding","expanding"], default=["sliding"])    
    purge_days   = st.number_input("Embargo/Purge (giorni)", 0, 365, 1, 1)
    max_configs  = st.number_input("Limite configurazioni", 1, 2000, 120, 1)
    line_opacity = st.slider("Opacità curve bundle", 0.05, 1.0, 0.25, 0.05)
show_band    = st.checkbox("Evidenzia banda robustezza (p10–p90) e mediana", True)

    st.divider()
    st.header("📏 Metrica di selezione/valutazione")
    metric = st.selectbox("Metrica", ["Sharpe", "Mean return", "Sortino", "Max Drawdown"])    
    ann    = st.number_input("Fattore annualizzazione", 1, 366, 252, 1)
    st.caption("Suggerimento: 252 ≈ giornaliera trading, 365 ≈ giornaliera calendario.")

# ============================
# Utils — Dati & Metriche
# ============================
@st.cache_data(show_spinner=False)
def generate_demo(n_strats: int, n_periods: int, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    data = {f"strategy_{i}": rng.normal(0.0, float(sigma), size=int(n_periods)) for i in range(int(n_strats))}
    idx = pd.date_range("2010-01-01", periods=int(n_periods), freq="D")
    return pd.DataFrame(data, index=idx)

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=0)
    # try parse index to datetime; if fails, keep numeric indexing
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    # numeric only
    df = df.apply(pd.to_numeric, errors="coerce")
    # drop all-NaN columns
    df = df.dropna(axis=1, how="all")
    return df

def infer_ann_from_index(idx: pd.Index) -> int:
    # Not cached: idx is unhashable for Streamlit caching. This is cheap anyway.
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        try:
            f = pd.infer_freq(idx)
            if f in ("B", "C"):
                return 252
            return 365
        except Exception:
            return 252
    return 252

# Metrics

def sharpe(mean, std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(std > 0, (mean / std) * np.sqrt(ann), -np.inf)
    return s

def sortino(mean, dn_std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(dn_std > 0, (mean / dn_std) * np.sqrt(ann), -np.inf)
    return s

# Vectorized per-column MDD

def series_mdd(s: pd.Series) -> float:
    if s.dropna().empty:
        return np.nan
    eq = (1.0 + s.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (peak - eq) / peak
    return float(dd.max())

def df_mdd_cols(df: pd.DataFrame) -> pd.Series:
    # compute MDD per column vectorized-like (loop in pandas but fast enough)
    eq = (1.0 + df.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (peak - eq) / peak
    return dd.max(axis=0)

# Build equity curve from sparse OOS (flat between OOS chunks, NaN before first OOS)

def equity_curve_from_oos(oos: pd.Series) -> pd.Series:
    eq = oos.fillna(0.0).cumsum()
    seen = oos.notna().cumsum() > 0
    return eq.where(seen)

# segment stats (mean, std, downside std)

def seg_stats(X: np.ndarray, a: int, b: int, cs: np.ndarray, css: np.ndarray):
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

# metric dispatcher

def metric_scores(metric: str, mean, std, dn_std, ann, mdd_vec=None):
    if metric == "Sharpe":       return sharpe(mean, std, ann)
    if metric == "Mean return":  return mean
    if metric == "Sortino":      return sortino(mean, dn_std, ann)
    if metric == "Max Drawdown": return -mdd_vec  # maximize negative MDD => minimize MDD
    raise ValueError("Metric not supported")

# ============================
# Splits — step = OOS (no overlap)
# ============================

def add_offset(dt: pd.Timestamp, unit: str, amount: float) -> pd.Timestamp:
    if unit == "giorni":
        return dt + pd.DateOffset(days=int(amount))
    if unit == "mesi":
        return dt + pd.DateOffset(months=int(amount))
    if unit == "anni":
        # allow floats (e.g., 2.5 anni) by mapping to months
        months = int(round(float(amount) * 12))
        return dt + pd.DateOffset(months=months)
    raise ValueError("unit must be 'giorni' | 'mesi' | 'anni'")


def build_wf_splits(index: pd.DatetimeIndex, start_date: pd.Timestamp,
                    is_amt: float, oos_amt: float, unit: str,
                    mode: str, purge_days: int) -> List[Tuple[int,int,int,int]]:
    """
    Return list of (is_start, is_end, oos_start, oos_end) **as positions**.
    step = length(OOS) in given unit (non-overlapping OOS by definition).
    Embargo/Purge excludes the gap from both IS and OOS (IS ends at the anchor; OOS starts after purge).
    """
    idx = pd.DatetimeIndex(index).sort_values()
    T = len(idx)
    if T == 0:
        return []

    # first OOS anchor: >= start_date and with >= is_amt behind
    min_is_end = add_offset(idx[0], unit="anni", amount=is_amt) if unit == "anni" else add_offset(idx[0], unit=unit, amount=is_amt)
    first_anchor_date = max(pd.to_datetime(start_date), min_is_end)
    anchor = idx.searchsorted(first_anchor_date, side="left")
    if anchor >= T:
        return []

    splits = []
    while True:
        # embargo: OOS starts after the gap, but IS ends at the anchor (not including embargo in IS)
        is_end_pos = anchor
        oos_s = anchor + int(purge_days)

        # define oos end by date offset
        if oos_s >= T:
            break
        oos_end_date = add_offset(idx[oos_s], unit=unit, amount=oos_amt)
        oos_e = idx.searchsorted(oos_end_date, side="left")
        if oos_e <= oos_s:
            oos_e = min(oos_s + 1, T)
        if oos_e > T:
            oos_e = T
        if oos_e > T:
            break

        if mode == "sliding":
            is_start_date = add_offset(idx[is_end_pos], unit=unit, amount=-is_amt)
            is_s = idx.searchsorted(is_start_date, side="left")
            is_e = is_end_pos
        elif mode == "expanding":
            is_s = 0
            is_e = is_end_pos
            # require at least is_amt in effect
            min_req = add_offset(idx[0], unit=unit, amount=is_amt)
            if idx[is_e - 1] < min_req:
                # advance anchor by OOS step and continue
                next_anchor_date = add_offset(idx[anchor], unit=unit, amount=oos_amt)
                anchor = idx.searchsorted(next_anchor_date, side="left")
                if anchor >= T:
                    break
                continue
        else:
            raise ValueError("mode must be 'sliding' or 'expanding'")

        if is_e - is_s >= 2 and oos_e - oos_s >= 1:
            splits.append((is_s, is_e, oos_s, oos_e))

        # step by OOS length (non-overlapping OOS)
        next_anchor_date = add_offset(idx[anchor], unit=unit, amount=oos_amt)
        anchor = idx.searchsorted(next_anchor_date, side="left")
        if anchor >= T:
            break

    return splits

# ============================
# WF per configurazione → serie OOS concatenata + stats
# ============================

def run_wf_config_concat_oos(data: pd.DataFrame, start_date, is_amt, oos_amt,
                             unit, mode, purge_days, metric, ann):
    X = data.to_numpy(copy=False)  # (T, N)
    T, N = X.shape
    cs  = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])
    css = np.vstack([np.zeros((1, N)), np.cumsum(X**2, axis=0)])
    splits = build_wf_splits(data.index, start_date, is_amt, oos_amt, unit, mode, purge_days)

    # concatenazione dei soli rendimenti OOS del vincitore per ogni split
    oos_concat = pd.Series(index=data.index, dtype=float)  # NaN dove non c'è OOS
    winners = []

    for (i0, i1, o0, o1) in splits:
        mean_is, std_is, dn_is = seg_stats(X, i0, i1, cs, css)

        if metric == "Max Drawdown":
            seg_is = data.iloc[i0:i1]
            mdd_vec = df_mdd_cols(seg_is).to_numpy(dtype=float)
            scores = metric_scores(metric, mean_is, std_is, dn_is, ann, mdd_vec=mdd_vec)
        else:
            scores = metric_scores(metric, mean_is, std_is, dn_is, ann)

        tmp = np.where(np.isfinite(scores), scores, -np.inf)
        if not np.isfinite(tmp).any() or np.all(tmp == -np.inf):
            # skip this split if no valid scores
            continue
        w_idx = int(np.argmax(tmp))
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
        stats["Max Drawdown"] = float(series_mdd(oos_ret))
        # CAGR on calendar span covered by OOS
        eq = (1.0 + oos_ret).cumprod()
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        stats["CAGR"] = float(eq.iloc[-1]**(1/years) - 1)
        stats["Hit-rate"] = float((oos_ret > 0).mean())
    else:
        for k in ["Mean return","Sharpe","Sortino","Max Drawdown","CAGR","Hit-rate"]:
            stats[k] = np.nan

    details = {
        "is": is_amt,
        "oos": oos_amt,
        "unit": unit,
        "mode": mode,
        "purge_days": purge_days,
        "splits": len(splits),
        "winner_last": winners[-1] if winners else None
    }
    return oos_concat, stats, details

# ============================
# Carica dati
# ============================
if uploaded is not None:
    data = load_csv(uploaded)
    st.success("✅ File caricato e normalizzato (numeric only, drop all-NaN cols)")
else:
    st.info("ℹ️ Nessun file caricato: uso dataset demo i.i.d. N(0, σ²).")
    data = generate_demo(n_strategies, n_periods, sigma_demo, seed_demo)
    st.caption(f"Demo seed = {seed_demo}")

# Optionally auto-set ann default on first load
if 'ann_initialized' not in st.session_state:
    st.session_state['ann_initialized'] = True
    ann_default = infer_ann_from_index(data.index)
    if ann == 252 and ann_default != 252:
        st.toast(f"Nota: frequenza rilevata → ann={ann_default}")

# ============================
# Anteprima DEMO — PnL CUMULATO (tutte le strategie)
# ============================
st.subheader("Anteprima — PnL cumulato (tutte le strategie)")

cum = data.cumsum()  # cumuliamo i ritorni → PnL che parte da 0
cum_long = (
    cum.reset_index()
       .melt("index", var_name="strategy", value_name="pnl")
       .rename(columns={"index": "date"})
)

chart_cum = (
    alt.Chart(cum_long)
    .mark_line(opacity=0.85)
    .encode(
        x=alt.X("date:T", title="Data"),
        y=alt.Y("pnl:Q", title="PnL cumulato (partenza 0)", scale=alt.Scale(zero=True)),
        color=alt.Color("strategy:N", title="Strategia", legend=None if data.shape[1] > 25 else alt.Legend()),
        tooltip=["date:T", "strategy:N", alt.Tooltip("pnl:Q", format=".4f")]
    )
    .properties(height=260)
    .interactive()
)
st.altair_chart(chart_cum, use_container_width=True)

# ============================
# Costruisci griglia configurazioni (step = OOS)
# ============================

def parse_list_nums(txt: str) -> List[float]:
    return [float(x) for x in re.split(r"[\s,;]+", str(txt).strip()) if x]

IS_list  = parse_list_nums(is_txt)
OOS_list = parse_list_nums(oos_txt)
modes    = modes_sel if modes_sel else ["sliding"]

grid = list(product(IS_list, OOS_list, modes))  # (IS, OOS, mode)
if len(grid) > max_configs:
    grid = grid[:max_configs]
    st.info(f"🔎 Configurazioni limitate a {max_configs} (aumenta 'Limite configurazioni' per elaborarne di più).")

st.write(f"**Configurazioni nel bundle:** {len(grid)} — (unità: {time_unit}, step = OOS, OOS non sovrapposti)")

# ============================
# Esegui il Bundle
# ============================

bundle_rows = []
equities: Dict[str, pd.Series] = {}
oos_concat_map: Dict[str, pd.Series] = {}

progress = st.progress(0.0, text="Elaborazione bundle…")

for k, (is_amt, oos_amt, mode) in enumerate(grid):
    oos_concat, stats, details = run_wf_config_concat_oos(
        data, start_date=pd.to_datetime(start_date),
        is_amt=is_amt, oos_amt=oos_amt, unit=time_unit,
        mode=mode, purge_days=int(purge_days),
        metric=metric, ann=int(ann)
    )
    key = f"IS={is_amt}{time_unit[0]} | OOS={oos_amt}{time_unit[0]} | {mode}"
    oos_concat_map[key] = oos_concat

    row = {"config": key, **stats, **details}
    bundle_rows.append(row)

    # PnL cumulato della sola serie OOS concatenata (gap = 0 return per visual)
    eq = oos_concat.fillna(0.0).cumsum()
    equities[key] = eq

    progress.progress((k + 1) / max(1, len(grid)), text=f"{k+1}/{len(grid)} configurazioni")

progress.empty()

# ============================
# Tabbed results (Bundle-first view)
# ============================

if len(bundle_rows) == 0:
    st.warning("Nessuna configurazione valida. Controlla range date, IS/OOS e unità.")
else:
    df_bundle = pd.DataFrame(bundle_rows).reset_index(drop=True)

    tab_bundle, tab_heatmap, tab_metrics, tab_downloads = st.tabs(["Bundle", "Heatmap", "Metriche & tabella", "Download"]) 

    with tab_bundle:
        st.subheader("📈 Bundle delle equity OOS concatenati — tutte le configurazioni")
        # Build equity curves for all configs
        if len(oos_concat_map) == 0:
            st.info("Nessuna equity da plottare.")
        else:
            eq_all = {k: equity_curve_from_oos(v) for k, v in oos_concat_map.items()}
            eq_df = pd.DataFrame(eq_all)
            eq_long = (
                eq_df.reset_index()
                     .melt("index", var_name="config", value_name="equity")
                     .rename(columns={"index": "date"})
            )

            # Robustness band (p10–p90) and median across configs per date
            band = None
            if show_band and not eq_df.empty:
                p10 = eq_df.quantile(0.10, axis=1, interpolation="linear")
                p50 = eq_df.quantile(0.50, axis=1, interpolation="linear")
                p90 = eq_df.quantile(0.90, axis=1, interpolation="linear")
                band_df = pd.DataFrame({"date": eq_df.index, "p10": p10.values, "p50": p50.values, "p90": p90.values}).dropna()
                band = (
                    alt.Chart(band_df)
                    .mark_area(opacity=0.25)
                    .encode(x="date:T", y="p10:Q", y2="p90:Q")
                ) + (
                    alt.Chart(band_df)
                    .mark_line()
                    .encode(x="date:T", y="p50:Q")
                )

            lines = (
                alt.Chart(eq_long.dropna())
                .mark_line(opacity=float(line_opacity))
                .encode(
                    x=alt.X("date:T", title="Data"),
                    y=alt.Y("equity:Q", title="PnL cumulato (solo OOS concatenati)", scale=alt.Scale(zero=True)),
                    color=alt.Color("config:N", legend=None),
                    tooltip=["date:T", "config:N", alt.Tooltip("equity:Q", format=".4f")]
                )
                .properties(height=420)
                .interactive()
            )

            chart = (band + lines) if band is not None else lines
            st.altair_chart(chart, use_container_width=True)

    with tab_heatmap:
        st.subheader("🗺️ Heatmap della metrica sulla griglia (per modalità selezionata)")
        if not df_bundle.empty:
            # build a compact df for heatmap
            hm = df_bundle.copy()
            hm["IS"] = [x.split("|")[0].split("=")[1].strip() for x in hm["config"]]
            hm["OOS"] = [x.split("|")[1].split("=")[1].strip() for x in hm["config"]]
            hm["Mode"] = hm["mode"].astype(str)
            metric_col = metric if metric in hm.columns else "Sharpe"
            hm[metric_col] = pd.to_numeric(hm[metric_col], errors="coerce")
            base = (
                alt.Chart(hm)
                .mark_rect()
                .encode(
                    x=alt.X("IS:N", title=f"IS ({time_unit})"),
                    y=alt.Y("OOS:N", title=f"OOS ({time_unit})"),
                    color=alt.Color(f"{metric_col}:Q", title=f"{metric_col}"),
                    tooltip=["config:N", alt.Tooltip(f"{metric_col}:Q", format=".3f"), "splits:N", "Mode:N"],
                )
                .properties(height=320)
            )
            modes_unique = hm["Mode"].unique().tolist()
            if len(modes_unique) > 1:
                ch = base.facet(column=alt.Column("Mode:N", title="Mode"))
            else:
                ch = base.properties(title=f"Mode: {modes_unique[0]}")
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Nessun dato per la heatmap.")

    with tab_metrics:
        st.subheader("Distribuzione metrica sul bundle")
        if not df_bundle.empty:
            metric_col = metric if metric in df_bundle.columns else "Sharpe"
            vals = pd.to_numeric(df_bundle[metric_col], errors="coerce").dropna()
            if not vals.empty:
                p5, p25, p50, p75, p95 = np.percentile(vals, [5,25,50,75,95])
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("p5", f"{p5:.3f}")
                c2.metric("p25", f"{p25:.3f}")
                c3.metric("median", f"{p50:.3f}")
                c4.metric("p75", f"{p75:.3f}")
                c5.metric("p95", f"{p95:.3f}")
                ch = (
                    alt.Chart(df_bundle[[metric_col]])
                    .mark_bar()
                    .encode(alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=40)), y="count()")
                    .properties(height=300)
                )
                st.altair_chart(ch, use_container_width=True)
        st.subheader("Tabella configurazioni (non ordinata per metrica)")
        st.dataframe(df_bundle, use_container_width=True)

    with tab_downloads:
        st.subheader("⬇️ Download risultati / OOS concatenati")
        st.download_button(
            "Scarica tabella configurazioni (CSV)",
            data=df_bundle.to_csv(index=False),
            file_name="wfb_bundle_summary.csv",
            mime="text/csv"
        )
        if not df_bundle.empty:
            out = pd.DataFrame({k: oos_concat_map[k] for k in df_bundle["config"]})
            st.download_button(
                "Scarica OOS concatenati (tutte le configurazioni, CSV)",
                data=out.to_csv(),
                file_name="wfb_oos_concat_all.csv",
                mime="text/csv"
            )
        else:
            st.info("Nessun OOS da scaricare.")
