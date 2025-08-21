# streamlit_app.py — Walk-Forward Bundle (v4-ultralight)
# Obiettivo: estetica "wow", massima leggerezza, messaggio chiaro sulla robustezza del bundle.
# - Tutte le linee OOS concatenate (nuvola grigio trasparente)
# - Banda p10–p90 + mediana (calcolate lato Pandas)
# - Evidenzia UNA configurazione (overlay)
# - Downsample duro (max_points) + resample display (settimanale/mensile)
# - Embargo corretto, IS/OOS in giorni/mesi/anni, MDD vettoriale
# - CSV loader robusto, progress bar, heatmap leggera, metrica distribuzione

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import product
from typing import List, Tuple, Dict
import re

# ============================
# Page / Global Style
# ============================
st.set_page_config(page_title="Walk-Forward Bundle — v4 ultralight", page_icon="📦", layout="wide")

# Minimal glassy look & feel
st.markdown("""
<style>
/* Base */
html, body, [class^="css"]  {
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji";
}
section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,.05); }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: 0.2px; }

/* Cards */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,.75), rgba(255,255,255,.60));
  backdrop-filter: blur(6px);
  border: 1px solid rgba(0,0,0,.06);
  border-radius: 16px;
  padding: 1rem 1.25rem;
  box-shadow: 0 8px 24px rgba(0,0,0,.06);
  margin-bottom: 1rem;
}

/* Subtle captions */
.small { font-size: 12px; color: rgba(0,0,0,.55); }

/* Hide footer watermark for cleanliness */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("### 🚶‍♂️📦 Walk-Forward Bundle — **robustezza visiva, zero fronzoli**")
st.caption("Stress test del modello al variare di IS/OOS/metriche. Focus: stabilità del **bundle** (non scegliere un vincitore).")

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
    st.caption("Se non carichi nulla: dataset demo i.i.d. N(0, σ²) dal 2010 in poi.")

    st.divider()
    st.header("🧭 Griglia bundle")
    start_date   = st.date_input("Inizio WFB", value=pd.to_datetime("2013-01-01"))
    time_unit    = st.radio("Unità IS/OOS", ["giorni", "mesi", "anni"], index=0, horizontal=True)
    is_txt       = st.text_input("Lista IS", value="3,5,8")
    oos_txt      = st.text_input("Lista OOS", value="63,126")
    modes_sel    = st.multiselect("Modalità", ["sliding","expanding"], default=["sliding"])
    purge_days   = st.number_input("Embargo/Purge (giorni)", 0, 365, 1, 1)
    max_configs  = st.number_input("Limite configurazioni", 1, 2000, 120, 1)

    st.divider()
    st.header("🎨 Grafico (ultralight)")
    show_band    = st.checkbox("Banda p10–p90 + mediana", True)
    display_freq = st.selectbox("Campionamento display", ["nessuno", "settimanale", "mensile"], index=0)
    max_points   = st.number_input("Max punti temporali (display)", 300, 20000, 2000, 100)
    cloud_alpha  = st.slider("Opacità nuvola bundle", 0.02, 0.5, 0.08, 0.01)

    st.divider()
    st.header("📏 Metrica selezione")
    metric = st.selectbox("Metrica", ["Sharpe", "Mean return", "Sortino", "Max Drawdown"])
    ann    = st.number_input("Fattore annualizzazione", 1, 366, 252, 1)
    st.caption("Tip: 252 ~ giorni di trading; 365 ~ calendario.")

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
    try: df.index = pd.to_datetime(df.index)
    except Exception: pass
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return df

def infer_ann_from_index(idx: pd.Index) -> int:
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        try:
            f = pd.infer_freq(idx)
            return 252 if f in ("B","C") else 365
        except Exception:
            return 252
    return 252

def sharpe(mean, std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(std > 0, (mean / std) * np.sqrt(ann), -np.inf)

def sortino(mean, dn_std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(dn_std > 0, (mean / dn_std) * np.sqrt(ann), -np.inf)

def series_mdd(s: pd.Series) -> float:
    if s.dropna().empty: return np.nan
    eq = (1.0 + s.fillna(0.0)).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    return float(dd.max())

def df_mdd_cols(df: pd.DataFrame) -> pd.Series:
    eq = (1.0 + df.fillna(0.0)).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    return dd.max(axis=0)

def equity_curve_from_oos(oos: pd.Series) -> pd.Series:
    eq = oos.fillna(0.0).cumsum()
    return eq.where(oos.notna().cumsum() > 0)

def resample_for_display(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex): return df
    rule = {"nessuno": None, "settimanale": "W", "mensile": "M"}[freq]
    if rule is None: return df
    try: return df.resample(rule).last()
    except Exception: return df

def decimate_by_stride(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if df.empty or max_points <= 0: return df
    if len(df) <= max_points: return df
    stride = int(np.ceil(len(df) / max_points))
    return df.iloc[::stride]

def seg_stats(X: np.ndarray, a: int, b: int, cs: np.ndarray, css: np.ndarray):
    n = b - a
    if n <= 1:
        N = X.shape[1]
        return (np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan))
    S  = cs[b]  - cs[a]
    SS = css[b] - css[a]
    mean = S / n
    var  = (SS - (S*S)/n) / (n - 1)
    std  = np.sqrt(np.maximum(var, 0.0))
    dn   = np.where(X[a:b] < 0.0, X[a:b], 0.0).std(axis=0, ddof=1)
    return mean, std, dn

def metric_scores(metric: str, mean, std, dn_std, ann, mdd_vec=None):
    if metric == "Sharpe":       return sharpe(mean, std, ann)
    if metric == "Mean return":  return mean
    if metric == "Sortino":      return sortino(mean, dn_std, ann)
    if metric == "Max Drawdown": return -mdd_vec
    raise ValueError("Metric not supported")

# ============================
# Splits — step = OOS (no overlap) con embargo
# ============================
def add_offset(dt: pd.Timestamp, unit: str, amount: float) -> pd.Timestamp:
    if unit == "giorni": return dt + pd.DateOffset(days=int(amount))
    if unit == "mesi":   return dt + pd.DateOffset(months=int(amount))
    if unit == "anni":   return dt + pd.DateOffset(months=int(round(float(amount)*12)))
    raise ValueError("unit must be 'giorni' | 'mesi' | 'anni'")

def build_wf_splits(index: pd.DatetimeIndex, start_date: pd.Timestamp,
                    is_amt: float, oos_amt: float, unit: str,
                    mode: str, purge_days: int) -> List[Tuple[int,int,int,int]]:
    idx = pd.DatetimeIndex(index).sort_values()
    T = len(idx)
    if T == 0: return []
    min_is_end = add_offset(idx[0], unit="anni", amount=is_amt) if unit=="anni" else add_offset(idx[0], unit=unit, amount=is_amt)
    first_anchor_date = max(pd.to_datetime(start_date), min_is_end)
    anchor = idx.searchsorted(first_anchor_date, side="left")
    if anchor >= T: return []
    splits = []
    while True:
        is_end_pos = anchor
        oos_s = anchor + int(purge_days)
        if oos_s >= T: break
        oos_end_date = add_offset(idx[oos_s], unit=unit, amount=oos_amt)
        oos_e = idx.searchsorted(oos_end_date, side="left")
        if oos_e <= oos_s: oos_e = min(oos_s + 1, T)
        if oos_e > T: oos_e = T
        if mode == "sliding":
            is_start_date = add_offset(idx[is_end_pos], unit=unit, amount=-is_amt)
            is_s = idx.searchsorted(is_start_date, side="left"); is_e = is_end_pos
        elif mode == "expanding":
            is_s = 0; is_e = is_end_pos
            min_req = add_offset(idx[0], unit=unit, amount=is_amt)
            if idx[is_e - 1] < min_req:
                next_anchor_date = add_offset(idx[anchor], unit=unit, amount=oos_amt)
                anchor = idx.searchsorted(next_anchor_date, side="left")
                if anchor >= T: break
                continue
        else:
            raise ValueError("mode must be 'sliding' or 'expanding'")
        if is_e - is_s >= 2 and oos_e - oos_s >= 1:
            splits.append((is_s, is_e, oos_s, oos_e))
        next_anchor_date = add_offset(idx[anchor], unit=unit, amount=oos_amt)
        anchor = idx.searchsorted(next_anchor_date, side="left")
        if anchor >= T: break
    return splits

# ============================
# WF per configurazione → OOS concatenata + stats
# ============================
def run_wf_config_concat_oos(data: pd.DataFrame, start_date, is_amt, oos_amt,
                             unit, mode, purge_days, metric, ann):
    X = data.to_numpy(copy=False)
    T, N = X.shape
    cs  = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])
    css = np.vstack([np.zeros((1, N)), np.cumsum(X**2, axis=0)])
    splits = build_wf_splits(data.index, start_date, is_amt, oos_amt, unit, mode, purge_days)

    oos_concat = pd.Series(index=data.index, dtype=float)
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
        if not np.isfinite(tmp).any() or np.all(tmp == -np.inf): continue
        w_idx = int(np.argmax(tmp))
        winners.append(data.columns[w_idx])
        oos_concat.iloc[o0:o1] = data.iloc[o0:o1, w_idx].to_numpy()

    oos_ret = oos_concat.dropna()
    stats = {}
    if not oos_ret.empty:
        mu = oos_ret.mean(); sd = oos_ret.std(ddof=1)
        dn = oos_ret.where(oos_ret < 0, 0.0).std(ddof=1)
        stats["Mean return"]  = float(mu)
        stats["Sharpe"]       = float((mu / sd) * np.sqrt(ann)) if sd > 0 else float("-inf")
        stats["Sortino"]      = float((mu / dn) * np.sqrt(ann)) if dn > 0 else float("-inf")
        stats["Max Drawdown"] = float(series_mdd(oos_ret))
        eq = (1.0 + oos_ret).cumprod()
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        stats["CAGR"] = float(eq.iloc[-1]**(1/years) - 1)
        stats["Hit-rate"] = float((oos_ret > 0).mean())
    else:
        for k in ["Mean return","Sharpe","Sortino","Max Drawdown","CAGR","Hit-rate"]:
            stats[k] = np.nan

    details = {"is": is_amt, "oos": oos_amt, "unit": unit, "mode": mode,
               "purge_days": purge_days, "splits": len(splits),
               "winner_last": winners[-1] if winners else None}
    return oos_concat, stats, details

# ============================
# Carica dati
# ============================
if uploaded is not None:
    data = load_csv(uploaded)
    st.success("✅ CSV caricato (numeric only, drop all-NaN cols)")
else:
    data = generate_demo(n_strategies, n_periods, sigma_demo, seed_demo)
    st.info("ℹ️ Dataset demo i.i.d. N(0, σ²) in uso.")
    st.caption(f"Demo seed = {seed_demo}")

if 'ann_initialized' not in st.session_state:
    st.session_state['ann_initialized'] = True
    ann_default = infer_ann_from_index(data.index)
    if ann == 252 and ann_default != 252:
        st.toast(f"Suggerimento: ann={ann_default} dalla frequenza indice")

# ============================
# Preview — PnL cumulato (tutte le strategie)
# ============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👀 Anteprima — PnL cumulato (tutte le strategie)")
cum = data.cumsum().reset_index().melt("index", var_name="strategy", value_name="pnl").rename(columns={"index": "date"})
chart_cum = (
    alt.Chart(cum)
    .mark_line(opacity=0.75)
    .encode(
        x=alt.X("date:T", title="Data"),
        y=alt.Y("pnl:Q", title="PnL cumulato (0 start)", scale=alt.Scale(zero=True)),
        color=alt.Color("strategy:N", legend=None if data.shape[1] > 20 else alt.Legend()),
    )
    .properties(height=220)
    .interactive()
)
st.altair_chart(chart_cum, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================
# Griglia configurazioni (step = OOS)
# ============================
def parse_list_nums(txt: str) -> List[float]:
    return [float(x) for x in re.split(r"[\s,;]+", str(txt).strip()) if x]

IS_list  = parse_list_nums(is_txt)
OOS_list = parse_list_nums(oos_txt)
modes    = modes_sel if modes_sel else ["sliding"]

grid = list(product(IS_list, OOS_list, modes))
if len(grid) > max_configs:
    grid = grid[:max_configs]
    st.info(f"🔎 Limitato a {max_configs} configurazioni (alza il limite per elaborarne di più).")

st.write(f"**Configurazioni nel bundle:** {len(grid)} — (unità: {time_unit}, step = OOS, OOS non sovrapposti)")

# ============================
# Esegui il Bundle
# ============================
bundle_rows = []
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
    bundle_rows.append({"config": key, **stats, **details})
    progress.progress((k + 1) / max(1, len(grid)), text=f"{k+1}/{len(grid)} config")
progress.empty()

if len(bundle_rows) == 0:
    st.warning("Nessuna configurazione valida. Controlla range date, IS/OOS e unità.")
    st.stop()

df_bundle = pd.DataFrame(bundle_rows).reset_index(drop=True)

# ============================
# TAB 1 — Bundle Plot (clean & stable)
# ============================
tab_bundle, tab_heatmap, tab_metrics, tab_downloads = st.tabs(
    ["Bundle", "Heatmap", "Metriche", "Download"]
)

with tab_bundle:
    st.subheader("📈 Bundle OOS concatenati")

    if len(oos_concat_map) == 0:
        st.info("Nessuna equity da plottare.")
    else:
        # 1) costruzione equity
        eq_all = {k: equity_curve_from_oos(v) for k, v in oos_concat_map.items()}
        eq_df = pd.DataFrame(eq_all)

        # 2) downsample max N punti (solo display)
        Nmax = 2000
        if len(eq_df) > Nmax:
            idx = np.linspace(0, len(eq_df) - 1, Nmax, dtype=int)
            eq_df = eq_df.iloc[idx]

        # 3) calcolo mediana
        median_curve = eq_df.median(axis=1)

        # 4) tutte le linee in grigio leggero
        cloud = (
            alt.Chart(eq_df.reset_index().melt("index", var_name="config", value_name="equity"))
            .mark_line(opacity=0.15, color="gray")
            .encode(x="index:T", y="equity:Q")
        )

        # 5) linea mediana in bianco
        median_line = (
            alt.Chart(median_curve.reset_index().rename(columns={"index": "date", 0: "equity"}))
            .mark_line(color="white", size=2)
            .encode(x="date:T", y="equity:Q")
        )

        # 6) Overlay opzionale: evidenzia UNA configurazione
        cfg_list = ["(nessuna)"] + list(eq_df.columns)
        chosen = st.selectbox("Evidenzia configurazione", cfg_list, index=0)

        highlight = None
        if chosen != "(nessuna)" and chosen in eq_df.columns:
            highlight = (
                alt.Chart(eq_df[[chosen]].reset_index().rename(columns={"index": "date", chosen: "equity"}))
                .mark_line(size=3, color="#FFD166")
                .encode(x="date:T", y="equity:Q")
            )

        # 7) composizione finale
        chart = cloud + median_line
        if highlight is not None:
            chart = chart + highlight

        st.altair_chart(chart.properties(height=460), use_container_width=True)




# ============================
# TAB 2 — Heatmap (leggera)
# ============================
with tab_heatmap:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🗺️ Heatmap metrica (IS×OOS) per modalità")
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
        )
        .properties(height=300)
    )
    modes_unique = hm["Mode"].unique().tolist()
    ch = base.facet(column=alt.Column("Mode:N", title="Mode")) if len(modes_unique) > 1 else base.properties(title=f"Mode: {modes_unique[0]}")
    st.altair_chart(ch, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# TAB 3 — Metriche (distribuzione)
# ============================
with tab_metrics:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Distribuzione della metrica sul bundle")
    metric_col = metric if metric in df_bundle.columns else "Sharpe"
    vals = pd.to_numeric(df_bundle[metric_col], errors="coerce").dropna()
    if not vals.empty:
        p5, p25, p50, p75, p95 = np.percentile(vals, [5,25,50,75,95])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("p5",  f"{p5:.3f}")
        c2.metric("p25", f"{p25:.3f}")
        c3.metric("mediana", f"{p50:.3f}")
        c4.metric("p75", f"{p75:.3f}")
        c5.metric("p95", f"{p95:.3f}")
        ch = (
            alt.Chart(df_bundle[[metric_col]])
            .mark_bar()
            .encode(alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=45)), y="count()")
            .properties(height=280)
        )
        st.altair_chart(ch, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================
# TAB 4 — Download
# ============================
with tab_downloads:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⬇️ Download")
    st.download_button(
        "Scarica tabella configurazioni (CSV)",
        data=df_bundle.to_csv(index=False),
        file_name="wfb_bundle_summary.csv",
        mime="text/csv"
    )
    out = pd.DataFrame({k: v for k, v in oos_concat_map.items()})
    st.download_button(
        "Scarica OOS concatenati (tutte le configurazioni, CSV)",
        data=out.to_csv(),
        file_name="wfb_oos_concat_all.csv",
        mime="text/csv"
    )
    st.markdown('<span class="small">I CSV sono in formato largo: colonne=Configurazioni, righe=timestamp.</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


