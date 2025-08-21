# =========================================================
# streamlit_app.py — Walk-Forward Bundle (v3) - Versione Funzionante
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from itertools import product
from typing import List, Tuple, Dict
import re
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import i0

# ============================
# Configurazione Pagina & Titolo
# ============================
st.set_page_config(page_title="🚀 WFB Bundle v3 — Funzionante", layout="wide")
st.title("🚶‍♂️📦 Walk-Forward Bundle — Concatenazione OOS (v3)")
st.markdown(
    """
Questa app esegue un **Walk-Forward Bundle (WFB)** con **step = OOS** (OOS non sovrapposti).
Per ogni configurazione (IS/OOS, metrica, modalità) selezioniamo **in-sample** la miglior strategia e
**concateniamo solo i rendimenti OOS** del vincitore creando **una serie** per configurazione.

L'obiettivo non è trovare un “vincitore”, ma valutare la **stabilità** sotto variazioni dei parametri.
Il grafico principale mostra **tutte** le equity OOS concatenate (una linea per configurazione) e, opzionale,
la **banda di robustezza (p10–p90)** con **mediana**.
"""
)

# ============================
# Sidebar — Dati & Griglia
# ============================
with st.sidebar:
    st.header("📦 Dati")
    n_strategies = st.number_input("N° strategie (demo)", 1, 500, 100, 1)
    n_periods    = st.number_input("N° periodi (demo)", 200, 100_000, 6000, 100)
    sigma_demo   = st.number_input("Vol demo σ", 0.0001, 0.5, 0.01, 0.0001, format="%.4f")
    seed_demo    = st.number_input("Seed demo", 0, 1_000_000, 42, 1)
    uploaded     = st.file_uploader("📂 Carica CSV (righe=periodi, colonne=strategie)", type=["csv"])
    st.caption("Se non carichi nulla, userò un dataset demo i.i.d. N(0, σ²) dal 2010 in poi.")

    st.divider()
    st.header("🧭 Griglia configurazioni")
    start_date   = st.date_input("Data di inizio WFB", value=pd.to_datetime("2013-01-01"))
    time_unit    = st.radio("Unità di misura IS/OOS", ["giorni", "mesi", "anni"], index=0, horizontal=True)
    is_txt       = st.text_input("Lista IS (unità sopra)", value="3,5,8")
    oos_txt      = st.text_input("Lista OOS (unità sopra)", value="63,126")
    modes_sel    = st.multiselect("Modalità", ["sliding","expanding"], default=["sliding"])
    purge_days   = st.number_input("Embargo/Purge (giorni)", 0, 365, 1, 1)
    max_configs  = st.number_input("Limite configurazioni", 1, 2000, 120, 1)

    st.divider()
    st.header("🎨 Grafico & Selezione")
    line_opacity = st.slider("Opacità curve bundle", 0.05, 1.0, 0.25, 0.05)
    show_band    = st.checkbox("Banda robustezza (p10–p90) + mediana", True)
    display_freq = st.selectbox("Campionamento grafico", ["nessuno", "settimanale", "mensile"], index=0)

    st.divider()
    st.header("📏 Metrica")
    metric = st.selectbox("Metrica di selezione/valutazione", ["Sharpe", "Mean return", "Sortino", "Max Drawdown"])
    ann    = st.number_input("Fattore annualizzazione", 1, 366, 252, 1)

# ============================
# Utils — Dati & Metriche (funzioni globali)
# ============================
def generate_demo(n_strats: int, n_periods: int, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    data = {f"strategy_{i}": rng.normal(0.0, float(sigma), size=int(n_periods)) for i in range(int(n_strats))}
    idx = pd.date_range("2010-01-01", periods=int(n_periods), freq="D")
    return pd.DataFrame(data, index=idx)

def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return df

def infer_ann_from_index(idx: pd.Index) -> int:
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        try:
            f = pd.infer_freq(idx)
            if f in ("B", "C"):
                return 252
            return 365
        except Exception:
            return 252
    return 252

def sharpe(mean, std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(std > 0, (mean / std) * np.sqrt(ann), -np.inf)
    return s

def sortino(mean, dn_std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(dn_std > 0, (mean / dn_std) * np.sqrt(ann), -np.inf)
    return s

def series_mdd(s: pd.Series) -> float:
    if s.dropna().empty:
        return np.nan
    eq = (1.0 + s.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (peak - eq) / peak
    return float(dd.max())

def df_mdd_cols(df: pd.DataFrame) -> pd.Series:
    eq = (1.0 + df.fillna(0.0)).cumprod()
    peak = eq.cummax()
    dd = (peak - eq) / peak
    return dd.max(axis=0)

def equity_curve_from_oos(oos: pd.Series) -> pd.Series:
    eq = oos.fillna(0.0).cumsum()
    seen = oos.notna().cumsum() > 0
    return eq.where(seen)

def resample_for_display(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    rule = {"nessuno": None, "settimanale": "W", "mensile": "M"}[freq]
    if rule is None:
        return df
    try:
        return df.resample(rule).last()
    except Exception:
        return df

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

def metric_scores(metric: str, mean, std, dn_std, ann, mdd_vec=None):
    if metric == "Sharpe":       return sharpe(mean, std, ann)
    if metric == "Mean return":  return mean
    if metric == "Sortino":      return sortino(mean, dn_std, ann)
    if metric == "Max Drawdown": return -mdd_vec  # minimizzare MDD ⇒ massimizzare -MDD
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
        months = int(round(float(amount) * 12))
        return dt + pd.DateOffset(months=months)
    raise ValueError("unit must be 'giorni' | 'mesi' | 'anni'")

def build_wf_splits(index: pd.DatetimeIndex, start_date: pd.Timestamp,
                    is_amt: float, oos_amt: float, unit: str,
                    mode: str, purge_days: int) -> List[Tuple[int,int,int,int]]:
    idx = pd.DatetimeIndex(index).sort_values()
    T = len(idx)
    if T == 0:
        return []

    min_is_end = add_offset(idx[0], unit=unit, amount=is_amt)
    first_anchor_date = max(pd.to_datetime(start_date), min_is_end)
    anchor = idx.searchsorted(first_anchor_date, side="left")
    if anchor >= T:
        return []

    splits = []
    while True:
        is_end_pos = anchor
        oos_s = anchor + int(purge_days)
        if oos_s >= T:
            break

        oos_end_date = add_offset(idx[oos_s], unit=unit, amount=oos_amt)
        oos_e = idx.searchsorted(oos_end_date, side="left")
        if oos_e <= oos_s:
            oos_e = min(oos_s + 1, T)
        if oos_e > T:
            oos_e = T

        if mode == "sliding":
            is_start_date = add_offset(idx[is_end_pos], unit=unit, amount=-is_amt)
            is_s = idx.searchsorted(is_start_date, side="left")
            is_e = is_end_pos
        elif mode == "expanding":
            is_s = 0
            is_e = is_end_pos
            min_req = add_offset(idx[0], unit=unit, amount=is_amt)
            if idx[is_e - 1] < min_req:
                next_anchor_date = add_offset(idx[anchor], unit=unit, amount=oos_amt)
                anchor = idx.searchsorted(next_anchor_date, side="left")
                if anchor >= T:
                    break
                continue
        else:
            raise ValueError("mode must be 'sliding' or 'expanding'")

        if is_e - is_s >= 2 and oos_e - oos_s >= 1:
            splits.append((is_s, is_e, oos_s, oos_e))

        next_anchor_date = add_offset(idx[anchor], unit=unit, amount=oos_amt)
        anchor = idx.searchsorted(next_anchor_date, side="left")
        if anchor >= T:
            break

    return splits

# ============================
# Funzione per l'esecuzione parallela
# ============================
def run_wf_config_concat_oos_wrapper(
    params: dict, data_buffer: io.BytesIO = None, data_is_demo: bool = False
):
    """
    Funzione wrapper per l'esecuzione in un processo separato.
    Accetta un buffer di dati o parametri per la demo per evitare problemi di serializzazione.
    """
    if data_is_demo:
        data = generate_demo(params['n_strategies'], params['n_periods'], params['sigma_demo'], params['seed_demo'])
    else:
        data = pd.read_csv(data_buffer, index_col=0)
        data.index = pd.to_datetime(data.index)

    return run_wf_config_concat_oos(
        data=data,
        start_date=pd.to_datetime(params['start_date']),
        is_amt=params['is_amt'],
        oos_amt=params['oos_amt'],
        unit=params['time_unit'],
        mode=params['mode'],
        purge_days=params['purge_days'],
        metric=params['metric'],
        ann=params['ann']
    )

def run_wf_config_concat_oos(data, start_date, is_amt, oos_amt, unit, mode, purge_days, metric, ann):
    # La tua funzione originale, ma ora non è più "globale"
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
        if not np.isfinite(tmp).any() or np.all(tmp == -np.inf):
            continue
        w_idx = int(np.argmax(tmp))
        winners.append(data.columns[w_idx])
        oos_concat.iloc[o0:o1] = data.iloc[o0:o1, w_idx].to_numpy()

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
# Logica Principale Streamlit
# ============================
# Carica o genera i dati una volta e li memorizza nella sessione
@st.cache_resource(show_spinner="Caricamento dati...")
def get_data(uploaded_file, n_strats, n_periods, sigma_demo, seed_demo):
    if uploaded_file is not None:
        data = load_csv(uploaded_file)
        file_info = uploaded_file.name
        st.success("✅ File caricato e normalizzato.")
    else:
        data = generate_demo(n_strats, n_periods, sigma_demo, seed_demo)
        file_info = "Dataset Demo"
        st.info("ℹ️ Nessun file caricato: uso dataset demo.")
    return data, file_info

data, file_info = get_data(uploaded, n_strategies, n_periods, sigma_demo, seed_demo)
st.markdown(f"**Dati in uso:** `{file_info}`")

if 'ann_initialized' not in st.session_state:
    st.session_state['ann_initialized'] = True
    ann_default = infer_ann_from_index(data.index)
    if ann == 252 and ann_default != 252:
        st.toast(f"Nota: frequenza rilevata → ann={ann_default}", icon="⚠️")

# ============================
# Anteprima
# ============================
st.subheader("Anteprima — PnL cumulato")
cum = data.cumsum()
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
        y=alt.Y("pnl:Q", title="PnL cumulato", scale=alt.Scale(zero=True)),
        color=alt.Color("strategy:N", title="Strategia", legend=None if data.shape[1] > 25 else alt.Legend()),
    )
    .properties(height=240)
    .interactive()
)
st.altair_chart(chart_cum, use_container_width=True)

# ============================
# Costruisci griglia & Esegui il Bundle
# ============================
def parse_list_nums(txt: str) -> List[float]:
    return [float(x) for x in re.split(r"[\s,;]+", str(txt).strip()) if x]

IS_list  = parse_list_nums(is_txt)
OOS_list = parse_list_nums(oos_txt)
modes    = modes_sel if modes_sel else ["sliding"]

grid = list(product(IS_list, OOS_list, modes))
if len(grid) > max_configs:
    grid = grid[:max_configs]
    st.info(f"🔎 Configurazioni limitate a {max_configs}.")

st.write(f"**Configurazioni nel bundle:** {len(grid)} — (unità: {time_unit}, step = OOS non sovrapposti)")

if st.button("▶️ Esegui Walk-Forward Bundle"):
    with st.spinner("🚀 Esecuzione in corso... Potrebbe volerci qualche istante..."):
        
        # Prepara i dati da passare ai processi figli
        data_buffer = io.BytesIO()
        data.to_csv(data_buffer, index=True)
        data_buffer.seek(0)
        
        tasks = []
        for is_amt, oos_amt, mode in grid:
            params = {
                'start_date': start_date.isoformat(),
                'is_amt': is_amt, 'oos_amt': oos_amt, 'time_unit': time_unit,
                'mode': mode, 'purge_days': int(purge_days), 'metric': metric, 'ann': int(ann),
                'n_strategies': n_strategies, 'n_periods': n_periods, 'sigma_demo': sigma_demo, 'seed_demo': seed_demo
            }
            tasks.append(params)
        
        bundle_rows = []
        oos_concat_map: Dict[str, pd.Series] = {}
        max_workers = os.cpu_count() or 4
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            if uploaded is not None:
                futures = {executor.submit(run_wf_config_concat_oos_wrapper, params, data_buffer=data_buffer): params for params in tasks}
            else:
                futures = {executor.submit(run_wf_config_concat_oos_wrapper, params, data_is_demo=True): params for params in tasks}
                
            for k, future in enumerate(as_completed(futures)):
                params = futures[future]
                try:
                    oos_concat, stats, details = future.result()
                    key = f"IS={details['is']}{time_unit[0]} | OOS={details['oos']}{time_unit[0]} | {details['mode']}"
                    oos_concat_map[key] = oos_concat
                    bundle_rows.append({"config": key, **stats, **details})
                except Exception as exc:
                    st.error(f"Errore nell'elaborazione della configurazione {params}: {exc}")
                    continue
                st.progress((k + 1) / max(1, len(grid)), text=f"Progresso: {k+1}/{len(grid)} configurazioni completate.")
        
        st.balloons()
        st.success("✅ Analisi completata!")

        if len(bundle_rows) == 0:
            st.warning("Nessuna configurazione valida. Controlla i parametri.")
        
        df_bundle = pd.DataFrame(bundle_rows).reset_index(drop=True)
        st.session_state['df_bundle'] = df_bundle
        st.session_state['oos_concat_map'] = oos_concat_map

# ============================
# Tabs per i Risultati (mostrate solo dopo l'esecuzione)
# ============================
if 'df_bundle' in st.session_state:
    df_bundle = st.session_state['df_bundle']
    oos_concat_map = st.session_state['oos_concat_map']

    tab_bundle, tab_heatmap, tab_metrics, tab_downloads = st.tabs(["Bundle", "Heatmap", "Metriche & Tabella", "Download"])

    # --- BUNDLE PLOT ---
    with tab_bundle:
        st.subheader("📈 Bundle delle equity OOS concatenate")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            cfg_list = ["(nessuna)"] + list(df_bundle["config"])
            chosen = st.selectbox("Evidenzia una configurazione", cfg_list, index=0)
            st.markdown("---")
            if chosen != "(nessuna)":
                st.caption(f"Dettagli: **{chosen}**")
                chosen_row = df_bundle[df_bundle["config"] == chosen].iloc[0]
                st.metric("Sharpe", f"{chosen_row['Sharpe']:.2f}")
                st.metric("CAGR", f"{chosen_row['CAGR']:.2%}")
                st.metric("Max Drawdown", f"{chosen_row['Max Drawdown']:.2%}")

        with col2:
            eq_all = {k: equity_curve_from_oos(v) for k, v in oos_concat_map.items()}
            eq_df = pd.DataFrame(eq_all)
            df_disp = resample_for_display(eq_df, display_freq)
            
            band = None
            if show_band and not df_disp.empty:
                p10 = df_disp.quantile(0.10, axis=1)
                p50 = df_disp.quantile(0.50, axis=1)
                p90 = df_disp.quantile(0.90, axis=1)
                band_df = pd.DataFrame({"date": df_disp.index, "p10": p10.values, "p50": p50.values, "p90": p90.values}).dropna()
                band = (
                    alt.Chart(band_df).mark_area(opacity=0.25).encode(x="date:T", y="p10:Q", y2="p90:Q", color=alt.value("lightblue"))
                ) + (
                    alt.Chart(band_df).mark_line().encode(x="date:T", y="p50:Q", color=alt.value("darkblue"), tooltip=[alt.Tooltip("p50", title="Mediana", format=".2f")])
                )

            eq_long = (
                df_disp.reset_index()
                        .melt("index", var_name="config", value_name="equity")
                        .rename(columns={"index": "date"})
            )
            lines = (
                alt.Chart(eq_long.dropna())
                .mark_line(opacity=float(line_opacity))
                .encode(
                    x=alt.X("date:T", title="Data"),
                    y=alt.Y("equity:Q", title="PnL cumulato (solo OOS)", scale=alt.Scale(zero=True)),
                    color=alt.Color("config:N", legend=None),
                    tooltip=[alt.Tooltip("config", title="Configurazione"), alt.Tooltip("equity", title="PnL", format=".2f")]
                )
                .properties(height=420)
                .interactive()
            )

            if chosen != "(nessuna)":
                hi = (
                    alt.Chart(
                        df_disp[[chosen]].rename(columns={chosen: "equity"})
                        .reset_index().rename(columns={"index": "date"})
                    )
                    .mark_line(size=3)
                    .encode(x="date:T", y=alt.Y("equity:Q"), color=alt.value("red"))
                )
                chart = (band + lines + hi) if band is not None else (lines + hi)
            else:
                chart = (band + lines) if band is not None else lines
            
            st.altair_chart(chart, use_container_width=True)

    # --- HEATMAP ---
    with tab_heatmap:
        st.subheader("🗺️ Heatmap della metrica sulla griglia")
        if not df_bundle.empty:
            hm = df_bundle.copy()
            hm["IS"] = [re.search(r"IS=(\d+\.?\d*)[a-z]", x).group(1) for x in hm["config"]]
            hm["OOS"] = [re.search(r"OOS=(\d+\.?\d*)[a-z]", x).group(1) for x in hm["config"]]
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
            ch = base.facet(column=alt.Column("Mode:N", title="Modalità")) if len(modes_unique) > 1 else base.properties(title=f"Modalità: {modes_unique[0]}")
            st.altair_chart(ch, use_container_width=True)
        else:
            st.info("Nessun dato per la heatmap.")

    # --- METRICHE & TABELLA ---
    with tab_metrics:
        st.subheader("Distribuzione metrica sul bundle")
        metric_col = metric if metric in df_bundle.columns else "Sharpe"
        vals = pd.to_numeric(df_bundle[metric_col], errors="coerce").dropna()
        if not vals.empty:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Mediana", f"{vals.median():.3f}"); c2.metric("Media", f"{vals.mean():.3f}")
            c3.metric("Max", f"{vals.max():.3f}"); c4.metric("Min", f"{vals.min():.3f}")
            c5.metric("Dev.Std.", f"{vals.std():.3f}")
            
            ch = alt.Chart(df_bundle[[metric_col]]).mark_bar().encode(
                alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=40)), y="count()"
            ).properties(height=300)
            st.altair_chart(ch, use_container_width=True)

        st.subheader("Tabella configurazioni")
        st.dataframe(df_bundle, use_container_width=True)

    # --- DOWNLOAD ---
    with tab_downloads:
        st.subheader("⬇️ Download risultati")
        st.download_button(
            "Scarica tabella configurazioni (CSV)",
            data=df_bundle.to_csv(index=False),
            file_name="wfb_bundle_summary.csv",
            mime="text/csv"
        )
        out = pd.DataFrame({k: oos_concat_map[k] for k in df_bundle["config"]})
        st.download_button(
            "Scarica OOS concatenati (tutte le configurazioni, CSV)",
            data=out.to_csv(),
            file_name="wfb_oos_concat_all.csv",
            mime="text/csv"
        )

