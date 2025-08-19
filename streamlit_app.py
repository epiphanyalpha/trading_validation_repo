# streamlit_app.py — PBO (CSCV) ready-to-run
from __future__ import annotations
import itertools, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="PBO (CSCV) — Trading Validation", layout="wide")
st.title("🔬 PBO (CSCV) — Test di Overfitting")
st.caption("Implementazione del Probability of Backtest Overfitting (López de Prado).")

# ------------- HELPERS: INPUT -------------
def read_csv_any(csv) -> pd.DataFrame:
    """Reads CSV and tries to detect 'date' column or index."""
    df = pd.read_csv(csv)
    # Normalize 'date'
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("date", "datetime", "timestamp"):
            date_col = c
            break
    if date_col is None:
        # try index named 'date'
        if "Unnamed: 0" in df.columns:
            # sometimes saved with index; keep as data column
            pass
    else:
        df[date_col] = pd.to_datetime(df[date_col], utc=False)
        df = df.set_index(date_col).sort_index()
    # If no date col found, try to parse first col as dates (best effort)
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.set_index(pd.to_datetime(df.iloc[:, 0], utc=False)).iloc[:, 1:]
            df = df.sort_index()
        except Exception:
            pass
    return df

def ensure_wide_returns(df: pd.DataFrame, data_kind: str, long_cols: Tuple[str,str,str]|None) -> pd.DataFrame:
    """
    Returns a WIDE DataFrame of periodic returns:
      index = DatetimeIndex, columns = strategies/configs, values = returns
    data_kind: 'returns' or 'equity'
    long_cols: if the CSV is long, pass (date_col, config_col, value_col)
    """
    if long_cols is not None:
        date_c, cfg_c, val_c = long_cols
        dfl = df.rename(columns={date_c:"date", cfg_c:"config", val_c:"value"}).copy()
        dfl["date"] = pd.to_datetime(dfl["date"], utc=False)
        dfl = dfl.dropna(subset=["date","config","value"])
        wide = dfl.pivot_table(index="date", columns="config", values="value", aggfunc="mean").sort_index()
    else:
        wide = df.copy()
        # if still has a 'date' column, make it index
        if "date" in [c.lower() for c in wide.columns]:
            real = {c.lower(): c for c in wide.columns}
            wide[real["date"]] = pd.to_datetime(wide[real["date"]], utc=False)
            wide = wide.set_index(real["date"]).sort_index()

    # Convert equity->returns if needed
    if data_kind == "equity":
        wide = wide.apply(lambda s: s.sort_index().pct_change(), axis=0)

    # Make sure values are numeric and drop all-null cols
    wide = wide.apply(pd.to_numeric, errors="coerce")
    wide = wide.dropna(how="all").fillna(0.0)  # treat missing as 0 ret by default
    return wide

# ------------- HELPERS: PBO ---------------
@dataclass
class PBOResult:
    S: int
    N_configs: int
    N_splits: int
    pbo: float
    lambdas: np.ndarray

def _make_blocks(dates: pd.DatetimeIndex, S: int) -> pd.Series:
    """Assign block id [0..S-1] to each date, in contiguous balanced slices."""
    ds = pd.DatetimeIndex(sorted(pd.unique(dates)))
    cut_idx = np.linspace(0, len(ds), S + 1, dtype=int)
    mapper: Dict[pd.Timestamp,int] = {}
    for b in range(S):
        for d in ds[cut_idx[b]:cut_idx[b+1]]:
            mapper[d] = b
    # return aligned series (same index order as input)
    return pd.Series([mapper[d] for d in dates], index=dates)

def compute_pbo_from_wide_returns(wide: pd.DataFrame, S: int = 10) -> Optional[PBOResult]:
    """
    wide: index=DatetimeIndex, columns=configs, values=periodic returns.
    Steps:
      1) Split into S contiguous blocks.
      2) For each comb of S/2 IS blocks (CSCV), pick best config in-sample.
      3) Compute OOS rank of that config; u = normalized rank; λ = logit(u).
      4) PBO = P(λ<0).
    """
    if wide.empty or wide.shape[1] < 2:
        st.warning("Servono almeno 2 configurazioni/strategie (2 colonne).")
        return None

    # 1) blocks
    wide = wide.sort_index()
    blocks = _make_blocks(wide.index, S)

    # 2) performance per (config, block): somma log(1+ret)
    logret = np.log1p(wide.astype(float))
    # aggregate by block: we need matrix (configs x blocks)
    perf_by_block = (
        logret.assign(_block=blocks.values)
              .groupby("_block")
              .sum()                  # sum over dates in block → per config
              .T                      # configs as rows
              .reindex(columns=sorted(perf_by_block.columns) if False else None)
    )
    # In case of missing blocks (edge cases), fill zeros
    # ensure exactly S columns (0..S-1)
    existing = set(perf_by_block.columns)
    for b in range(S):
        if b not in existing:
            perf_by_block[b] = 0.0
    perf_by_block = perf_by_block.reindex(columns=sorted(perf_by_block.columns))

    cfg_names = list(perf_by_block.index)
    block_ids  = list(perf_by_block.columns)

    if len(block_ids) < 2:
        st.warning("Troppi pochi blocchi per CSCV.")
        return None

    k = len(block_ids) // 2  # IS block count
    if k == 0:
        st.warning("S deve essere ≥ 2.")
        return None

    combs = list(itertools.combinations(block_ids, k))
    M = perf_by_block.values  # shape: (N_cfg, S)

    lambdas: List[float] = []
    for is_blocks in combs:
        is_mask  = np.isin(block_ids, is_blocks)
        oos_mask = ~is_mask

        is_perf   = M[:, is_mask].mean(axis=1)    # mean IS perf per config
        oos_all   = M[:, oos_mask].mean(axis=1)   # mean OOS perf per config

        j_star = int(np.argmax(is_perf))          # argmax in IS

        # rank OOS (1=worst, N=best)
        order = np.argsort(oos_all)               # ascending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(oos_all) + 1)

        u = (ranks[j_star] - 1) / (len(oos_all) - 1)
        u = float(np.clip(u, 1e-9, 1-1e-9))       # stabilize
        lam = math.log(u / (1 - u))
        lambdas.append(lam)

    lambdas = np.array(lambdas, dtype=float)
    pbo = float(np.mean(lambdas < 0.0))
    return PBOResult(S=S, N_configs=wide.shape[1], N_splits=len(combs), pbo=pbo, lambdas=lambdas)

# ------------- SIDEBAR INPUT --------------
st.sidebar.header("📥 Dati")
fmt = st.sidebar.selectbox("Formato dati nel CSV", ["Wide (date + colonne per strategia)", "Long (date, config, value)"])
kind = st.sidebar.selectbox("Le serie sono", ["Returns (periodici)", "Equity / P&L (cumulato)"])
S = st.sidebar.slider("S = numero di blocchi CSCV", min_value=4, max_value=20, step=2, value=10)

uploaded = st.sidebar.file_uploader("Carica CSV", type=["csv"])

# ------------- LOAD & PREP DATA -----------
wide_returns: Optional[pd.DataFrame] = None
if uploaded is not None:
    df = read_csv_any(uploaded)
    if fmt.startswith("Long"):
        # ask for column names quickly (simple guess)
        cols = [c for c in df.columns]
        # naive defaults
        date_c = next((c for c in cols if str(c).lower() in ("date","datetime")), cols[0])
        cfg_c  = "config" if "config" in [c.lower() for c in cols] else cols[1]
        val_c  = "value" if "value"  in [c.lower() for c in cols] else cols[2]
        wide_returns = ensure_wide_returns(df, "equity" if kind.startswith("Equity") else "returns",
                                           (date_c, cfg_c, val_c))
    else:
        wide_returns = ensure_wide_returns(df, "equity" if kind.startswith("Equity") else "returns", None)

# ------------- RESULTS: PBO ---------------
tab_pbo, tab_preview = st.tabs(["✅ Risultati PBO", "🗂️ Anteprima dati"])

with tab_preview:
    if wide_returns is None:
        st.info("Carica un CSV per vedere l’anteprima.")
    else:
        st.subheader("Anteprima (prime righe) — Wide returns")
        st.write(wide_returns.head())
        st.write(f"Righe: {len(wide_returns):,}  •  Configurazioni: {wide_returns.shape[1]:,}")

with tab_pbo:
    st.subheader("Probability of Backtest Overfitting (CSCV)")
    if wide_returns is None:
        st.warning("Carica un CSV e imposta il formato.")
    elif wide_returns.shape[1] < 2:
        st.warning("Servono almeno 2 colonne (2 strategie/configurazioni).")
    else:
        res = compute_pbo_from_wide_returns(wide_returns, S=S)
        if res is None:
            st.stop()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("S (blocchi)", f"{res.S}")
        c2.metric("# Configurazioni", f"{res.N_configs}")
        c3.metric("# Split CSCV", f"{res.N_splits}")
        c4.metric("PBO", f"{res.pbo:.2%}")

        # Histogram of lambda
        lam_df = pd.DataFrame({"lambda": res.lambdas})
        hist = (
            alt.Chart(lam_df)
            .mark_bar()
            .encode(x=alt.X("lambda:Q", bin=alt.Bin(maxbins=40)), y="count()")
            .properties(height=280, title="Distribuzione di λ = logit(u)")
        )
        st.altair_chart(hist, use_container_width=True)

        # CDF of u
        u = 1 / (1 + np.exp(-res.lambdas))
        cdf_df = pd.DataFrame({
            "u": np.sort(u),
            "F(u)": np.linspace(0, 1, len(u), endpoint=True)
        })
        line = (
            alt.Chart(cdf_df)
            .mark_line()
            .encode(x=alt.X("u:Q", scale=alt.Scale(domain=[0,1])), y=alt.Y("F(u):Q"))
            .properties(height=280, title="CDF di u (rank normalizzato OOS)")
        )
        st.altair_chart(line, use_container_width=True)

        st.caption(
            "Interpretazione: **PBO** è la probabilità che la configurazione selezionata in IS "
            "performi **sotto la mediana** in OOS. Valori alti → maggiore rischio di overfitting."
        )
