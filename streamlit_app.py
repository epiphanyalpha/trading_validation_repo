# ============================
# streamlit_app.py
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re  # keep for parsing lists
from typing import List, Tuple, Dict

st.set_page_config(
    page_title="Walk-Forward Bundle Validator",
    layout="wide"
)

# ---------- Small utility: cached file loader ----------
@st.cache_resource(show_spinner=False)
def load_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# -------------------------------------------------
# Altair: light theme, hide toolbar, no interactivity
# -------------------------------------------------
alt.data_transformers.disable_max_rows()          # avoid silent downsampling
alt.renderers.set_embed_options(actions=False)    # hide Vega toolbar

def _wfb_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {
                "labelFontSize": 12,
                "titleFontSize": 12,
                "grid": True,
                "gridOpacity": 0.15
            },
            "legend": {"labelFontSize": 11, "titleFontSize": 11}
        }
    }
alt.themes.register("wfb_light", _wfb_theme)
alt.themes.enable("wfb_light")

# -------------------------------------------------
# Optional tiny CSS touch (static, very lightweight)
# -------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Intro
# ----------------------------
st.title("⚡ Walk-Forward Bundle Validator")

st.markdown(
    """
**What this app does**

This tool validates a trading strategy with a *bundle* of walk-forward analyses.
Instead of relying on a single IS/OOS split, it runs **many configurations** (different
IS/OOS lengths, modes, and selection metrics), concatenates the **out-of-sample (OOS)**
returns from each configuration, and compares the resulting equity curves.

If *many* distinct walk-forward configurations deliver stable OOS performance, that's strong
evidence your strategy is **robust** rather than overfit.

**Key ideas**
- *Walk-Forward:* repeatedly train on **IS (in-sample)** and test on **OOS (out-of-sample)**.
- *Bundle:* run *multiple* walk-forward setups at once (various IS/OOS horizons, metrics).
- *Robustness:* consistency across the bundle matters more than a single best setting.

**How to use**
1. Upload your strategies’ return series (rows = time, columns = strategies) or use the demo data.
2. Define a grid of walk-forward configurations (IS/OOS units and lengths, mode, purge).
3. Review:
   - **Bundle equity**: concatenated OOS PnL for all configs (all lines, static).
   - **Heatmap** of the selection metric across IS×OOS (by mode).
   - **Distribution** of the chosen metric across the bundle.
4. **Download** the summary and the concatenated OOS series.
"""
)

# ----------------------------
# Demo schema helpers (used below)
# ----------------------------
def make_walkforward_schema(n_is=3, n_oos=3, n_blocks=3):
    """Simple dataframe with IS (green) and OOS (red) blocks to illustrate the idea."""
    data = []
    for block in range(n_blocks):
        for i in range(n_is):
            data.append({"step": block, "pos": i, "type": "IS"})
        for j in range(n_oos):
            data.append({"step": block, "pos": n_is + j, "type": "OOS"})
    return pd.DataFrame(data)

def render_walkforward_schema():
    df = make_walkforward_schema(n_is=3, n_oos=3, n_blocks=4)
    chart = (
        alt.Chart(df)
        .mark_rect(stroke="black", strokeWidth=0.5)
        .encode(
            x=alt.X("pos:O", title=None),
            y=alt.Y("step:O", title="Walk-Forward step"),
            color=alt.Color(
                "type:N",
                scale=alt.Scale(domain=["IS", "OOS"], range=["#06D6A0", "#EF476F"]),
                legend=alt.Legend(title="Block")
            )
        )
        .properties(width=300, height=200)
    )
    st.altair_chart(chart, use_container_width=False)

st.divider()

# ============================
# Core functions + Sidebar + Load data + Bundle
# ============================

# ---- Config & caching helpers ----
@st.cache_data(show_spinner=False)
def generate_demo(n_strats: int, n_periods: int, sigma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    data = {f"strategy_{i}": rng.normal(0.0, float(sigma), size=int(n_periods)) for i in range(int(n_strats))}
    idx = pd.date_range("2010-01-01", periods=int(n_periods), freq="D")
    return pd.DataFrame(data, index=idx)

@st.cache_data(show_spinner=False)
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
            return 252 if f in ("B","C") else 365
        except Exception:
            return 252
    return 252

# ---- Metrics ----
def sharpe(mean, std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(std > 0, (mean / std) * np.sqrt(ann), -np.inf)

def sortino(mean, dn_std, ann):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(dn_std > 0, (mean / dn_std) * np.sqrt(ann), -np.inf)

def series_mdd(s: pd.Series) -> float:
    if s.dropna().empty:
        return np.nan
    eq = (1.0 + s.fillna(0.0)).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    return float(dd.max())

def df_mdd_cols(df: pd.DataFrame) -> pd.Series:
    eq = (1.0 + df.fillna(0.0)).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    return dd.max(axis=0)

# ---- Equity helpers (PnL-like cumsum, display-only resampling/decimation) ----
def equity_curve_from_oos(oos: pd.Series) -> pd.Series:
    """PnL-like cumulative (starts at 0), robust to gaps."""
    return oos.fillna(0.0).cumsum()

def resample_for_display(df: pd.DataFrame, freq_label_internal: str) -> pd.DataFrame:
    """freq_label_internal is one of: 'nessuno', 'settimanale', 'mensile'."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    rule = {"nessuno": None, "settimanale": "W", "mensile": "M"}[freq_label_internal]
    if rule is None:
        return df
    try:
        return df.resample(rule).last()
    except Exception:
        return df

def decimate_by_stride(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if df.empty or max_points <= 0:
        return df
    if len(df) <= max_points:
        return df
    stride = int(np.ceil(len(df) / max_points))
    return df.iloc[::stride]

# ---- Selection segment stats ----
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

# ---- Date offsets (internal expects Italian: 'giorni' | 'mesi' | 'anni') ----
def add_offset(dt: pd.Timestamp, unit: str, amount: float) -> pd.Timestamp:
    if unit == "giorni":
        return dt + pd.DateOffset(days=int(amount))
    if unit == "mesi":
        return dt + pd.DateOffset(months=int(amount))
    if unit == "anni":
        return dt + pd.DateOffset(months=int(round(float(amount) * 12)))
    raise ValueError("unit must be 'giorni' | 'mesi' | 'anni'")

# ---- Build splits (step = OOS; non-overlapping OOS; embargo) ----
def build_wf_splits(index: pd.DatetimeIndex, start_date: pd.Timestamp,
                    is_amt: float, oos_amt: float,
                    is_unit: str, oos_unit: str,
                    mode: str, purge_days: int) -> List[Tuple[int,int,int,int]]:
    idx = pd.DatetimeIndex(index).sort_values()
    T = len(idx)
    if T == 0:
        return []

    # Require at least IS amount before first anchor
    min_is_end = add_offset(idx[0], unit=is_unit, amount=is_amt)
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

        # OOS end from oos_unit
        oos_end_date = add_offset(idx[oos_s], unit=oos_unit, amount=oos_amt)
        oos_e = idx.searchsorted(oos_end_date, side="left")
        if oos_e <= oos_s:
            oos_e = min(oos_s + 1, T)
        if oos_e > T:
            oos_e = T

        if mode == "sliding":
            is_start_date = add_offset(idx[is_end_pos], unit=is_unit, amount=-is_amt)
            is_s = idx.searchsorted(is_start_date, side="left")
            is_e = is_end_pos
        elif mode == "expanding":
            is_s = 0
            is_e = is_end_pos
            # Need at least is_amt before starting
            min_req = add_offset(idx[0], unit=is_unit, amount=is_amt)
            if idx[is_e - 1] < min_req:
                next_anchor_date = add_offset(idx[anchor], unit=oos_unit, amount=oos_amt)
                anchor = idx.searchsorted(next_anchor_date, side="left")
                if anchor >= T:
                    break
                continue
        else:
            raise ValueError("mode must be 'sliding' or 'expanding'")

        if is_e - is_s >= 2 and oos_e - oos_s >= 1:
            splits.append((is_s, is_e, oos_s, oos_e))

        # Step forward by OOS (in oos_unit)
        next_anchor_date = add_offset(idx[anchor], unit=oos_unit, amount=oos_amt)
        anchor = idx.searchsorted(next_anchor_date, side="left")
        if anchor >= T:
            break

    return splits

# ---- Run one WF config → concatenated OOS + stats ----
def run_wf_config_concat_oos(data: pd.DataFrame, start_date, is_amt, oos_amt,
                             is_unit, oos_unit, mode, purge_days, metric, ann):
    X = data.to_numpy(copy=False)
    T, N = X.shape
    cs  = np.vstack([np.zeros((1, N)), np.cumsum(X, axis=0)])
    css = np.vstack([np.zeros((1, N)), np.cumsum(X**2, axis=0)])

    splits = build_wf_splits(
        data.index, start_date, is_amt, oos_amt,
        is_unit, oos_unit, mode, purge_days
    )

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

        # append only the OOS of the winner
        oos_concat.iloc[o0:o1] = data.iloc[o0:o1, w_idx].to_numpy()

    # Stats on concatenated OOS
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
        # Calendar CAGR over the OOS span
        eq = (1.0 + oos_ret).cumprod()
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        stats["CAGR"] = float(eq.iloc[-1]**(1/years) - 1)
        stats["Hit-rate"] = float((oos_ret > 0).mean())
    else:
        for k in ["Mean return","Sharpe","Sortino","Max Drawdown","CAGR","Hit-rate"]:
            stats[k] = np.nan

    details = {
        "is": is_amt, "is_unit": is_unit,
        "oos": oos_amt, "oos_unit": oos_unit,
        "mode": mode, "purge_days": purge_days, "splits": len(splits),
        "winner_last": winners[-1] if winners else None
    }
    return oos_concat, stats, details

# ----------------------------
# Sidebar — data & grid (English UI → internal tokens)
# ----------------------------
# Mappings: English UI -> internal Italian tokens (so core functions remain unchanged)
UNITS_EN_TO_IT = {"days": "giorni", "months": "mesi", "years": "anni"}
UNIT_ABBR_EN   = {"days": "d",      "months": "m",   "years": "y"}
DISPLAY_EN_TO_IT = {"none": "nessuno", "weekly": "settimanale", "monthly": "mensile"}

with st.sidebar:
    st.header("📦 Data")
    n_strategies = st.number_input("Number of strategies (demo)", 1, 500, 12, 1)
    n_periods    = st.number_input("Number of periods (demo)", 200, 100_000, 6000, 100)
    sigma_demo   = st.number_input("Volatility σ (demo)", 0.0001, 0.5, 0.01, 0.0001, format="%.4f")
    seed_demo    = st.number_input("Demo seed", 0, 1_000_000, 42, 1)
    uploaded     = st.file_uploader("📂 CSV (rows=time, columns=strategies)", type=["csv"])
    st.caption("If nothing is uploaded: demo i.i.d. N(0, σ²) data from 2010 onward.")

    st.divider()
    st.header("🧭 Bundle grid")
    start_date    = st.date_input("Start date (WFB)", value=pd.to_datetime("2013-01-01"))

    # English options; convert to internal strings for logic
    is_unit_en  = st.radio("IS unit",  ["years", "months", "days"], index=0, horizontal=True)
    oos_unit_en = st.radio("OOS unit", ["days", "months", "years"], index=0, horizontal=True)
    is_unit  = UNITS_EN_TO_IT[is_unit_en]
    oos_unit = UNITS_EN_TO_IT[oos_unit_en]

    is_txt    = st.text_input("IS list (numbers; in IS unit)",  value="3,5,8")
    oos_txt   = st.text_input("OOS list (numbers; in OOS unit)", value="63,126")
    modes_sel = st.multiselect("Mode", ["sliding","expanding"], default=["sliding"])
    purge_days= st.number_input("Embargo/Purge (days)", 0, 365, 1, 1)
    max_configs = st.number_input("Config limit", 1, 2000, 120, 1)

    st.divider()
    st.header("🎨 Chart")
    display_freq_en = st.selectbox("Display sampling", ["none", "weekly", "monthly"], index=0)
    display_freq_internal = DISPLAY_EN_TO_IT[display_freq_en]
    max_points   = st.number_input("Max time points (display)", 300, 20000, 2000, 100)

    st.divider()
    st.header("📏 Selection metric")
    metric = st.selectbox("Metric", ["Sharpe", "Mean return", "Sortino", "Max Drawdown"])
    ann    = st.number_input("Annualization factor", 1, 366, 252, 1)
    st.caption("Tip: 252 ~ trading days; 365 ~ calendar days.")

# ----------------------------
# Load data
# ----------------------------
if uploaded is not None:
    data = load_csv(uploaded)
    st.success("✅ CSV loaded (numeric only, dropped all-NaN columns).")
else:
    data = generate_demo(n_strategies, n_periods, sigma_demo, seed_demo)
    st.info("ℹ️ Using demo i.i.d. N(0, σ²) dataset.")
    st.caption(f"Demo seed = {seed_demo}")

if 'ann_initialized' not in st.session_state:
    st.session_state['ann_initialized'] = True
    ann_default = infer_ann_from_index(data.index)
    if ann == 252 and ann_default != 252:
        st.toast(f"Suggestion: ann={ann_default} inferred from index frequency")

# ----------------------------
# Visual WF schematic (large & legible)
# ----------------------------
def parse_list_nums(txt: str):
    return [float(x) for x in re.split(r"[\s,;]+", str(txt).strip()) if x]

def wf_schematic(start_date, is_unit, oos_unit, is_amt, oos_amt, purge_days=1, splits=4) -> alt.Chart:
    def _off(dt, unit, amt):
        return add_offset(dt, unit=unit, amount=amt)

    rows = []
    base_start = pd.to_datetime(start_date)
    anchor = max(_off(base_start, is_unit, is_amt), base_start)

    for k in range(1, splits + 1):
        is_end   = anchor
        is_start = _off(is_end, is_unit, -is_amt)
        oos_start= is_end + pd.DateOffset(days=int(purge_days))
        oos_end  = _off(oos_start, oos_unit, oos_amt)
        rows += [
            {"split": k, "phase": "IS",      "start": is_start, "end": is_end},
            {"split": k, "phase": "Embargo", "start": is_end,   "end": oos_start},
            {"split": k, "phase": "OOS",     "start": oos_start,"end": oos_end},
        ]
        anchor = _off(anchor, oos_unit, oos_amt)  # step = OOS

    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadius=3)  # subtle rounding, cheap
        .encode(
            x=alt.X("start:T", title="Time"),
            x2="end:T",
            y=alt.Y("split:O", title="Split", sort="ascending",
                    scale=alt.Scale(padding=0.25)),
            color=alt.Color(
                "phase:N",
                scale=alt.Scale(
                    domain=["IS","Embargo","OOS"],
                    range=["#27AE60","#CFCFCF","#E74C3C"]
                ),
                legend=alt.Legend(title=None, orient="top")
            ),
        )
        .properties(height=260)
    )

st.markdown("#### Visual schema: IS → Embargo → OOS (step = OOS)")
is_first  = parse_list_nums(is_txt)[0] if parse_list_nums(is_txt) else 3.0
oos_first = parse_list_nums(oos_txt)[0] if parse_list_nums(oos_txt) else 63.0
st.altair_chart(
    wf_schematic(start_date, is_unit, oos_unit, is_first, oos_first, int(purge_days), splits=4),
    use_container_width=True
)
st.caption("Only the first 4 steps are shown for illustration. Calculations use the full series.")

st.markdown("#### 🎞️ Bundle: multiple walk-forwards at a glance")
try:
    video_bytes = load_bytes("1750240471108.mp4")   # file placed in the repo root
    st.video(video_bytes)
    st.caption("In the bundle we combine OOS returns from multiple configurations (animated illustration).")
except FileNotFoundError:
    st.warning("Video not found. Make sure 1750240471108.mp4 is in the same folder as streamlit_app.py.")

st.divider()

# ----------------------------
# Preview — cumulative PnL (all strategies)
# ----------------------------
st.subheader("👀 Preview — Cumulative PnL (all strategies)")
cum = data.cumsum()
cum.index.name = "date"
st.line_chart(cum, height=240, use_container_width=True)

# ----------------------------
# Build configuration grid
# ----------------------------
IS_list  = parse_list_nums(is_txt)
OOS_list = parse_list_nums(oos_txt)
modes    = modes_sel if modes_sel else ["sliding"]

from itertools import product
grid = list(product(IS_list, OOS_list, modes))
if len(grid) > max_configs:
    grid = grid[:max_configs]
    st.info(f"🔎 Limited to {max_configs} configurations (increase the limit to run more).")

# For display in the text below, we’ll show English unit abbreviations
is_abbr_en  = UNIT_ABBR_EN[is_unit_en]
oos_abbr_en = UNIT_ABBR_EN[oos_unit_en]

st.write(f"**Configurations in bundle:** {len(grid)} — (IS in {is_unit_en}, OOS in {oos_unit_en}, step = OOS, non-overlapping)")

# ----------------------------
# Run the Bundle
# ----------------------------
bundle_rows = []
oos_concat_map: Dict[str, pd.Series] = {}

progress = st.progress(0.0, text="Running bundle…")
for k, (is_amt, oos_amt, mode) in enumerate(grid):
    oos_concat, stats, details = run_wf_config_concat_oos(
        data, start_date=pd.to_datetime(start_date),
        is_amt=is_amt, oos_amt=oos_amt,
        is_unit=is_unit, oos_unit=oos_unit,  # internal tokens
        mode=mode, purge_days=int(purge_days),
        metric=metric, ann=int(ann)
    )
    key = f"IS={is_amt}{is_abbr_en} | OOS={oos_amt}{oos_abbr_en} | {mode}"
    oos_concat_map[key] = oos_concat
    bundle_rows.append({"config": key, **stats, **details})
    progress.progress((k + 1) / max(1, len(grid)), text=f"{k+1}/{len(grid)} configs")
progress.empty()

if len(bundle_rows) == 0:
    st.warning("No valid configuration. Check date range, IS/OOS lengths and units.")
    st.stop()

df_bundle = pd.DataFrame(bundle_rows).reset_index(drop=True)

# ============================
# Tabs: Bundle, Heatmap, Metrics, Download
# ============================
tab_bundle, tab_heatmap, tab_metrics, tab_downloads = st.tabs(
    ["Bundle", "Heatmap", "Metrics", "Download"]
)

# ----------------------------
# TAB 1 — Bundle (all lines, static, lightweight)
# ----------------------------
with tab_bundle:
    st.subheader("📈 Concatenated OOS Bundle — all configurations")

    if len(oos_concat_map) == 0:
        st.info("No equity to plot.")
    else:
        # cumulative (0-based) for all configurations
        eq_all = {k: oos_concat_map[k].fillna(0.0).cumsum() for k in oos_concat_map.keys()}
        eq_df = pd.DataFrame(eq_all)
        eq_df.index.name = "date"

        # display-only resample/decimate
        eq_df = resample_for_display(eq_df, display_freq_internal)
        if len(eq_df) > int(max_points):
            eq_df = decimate_by_stride(eq_df, int(max_points))

        # long format; NO legend, NO tooltips, single neutral stroke
        plot_df = eq_df.reset_index().melt("date", var_name="config", value_name="equity")

        chart = (
            alt.Chart(plot_df)
            .mark_line(strokeWidth=0.7, opacity=0.55, clip=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("equity:Q", title="Cumulative PnL (concatenated OOS only)", scale=alt.Scale(zero=True)),
                detail="config:N"   # separate paths without heavy color scale/legend
            )
            .properties(height=420)
        )

        st.altair_chart(chart, use_container_width=True)

# ----------------------------
# TAB 2 — Heatmap (IS×OOS) by mode — static, clean
# ----------------------------
with tab_heatmap:
    st.subheader("🗺️ Metric heatmap (IS × OOS)")

    hm = df_bundle.copy()
    # Extract IS/OOS labels from "config"
    try:
        hm["IS"]  = hm["config"].str.extract(r"IS=([^\|]+)\|")[0].str.strip().str.replace("IS=", "", regex=False)
        hm["OOS"] = hm["config"].str.extract(r"OOS=([^\|]+)\|")[0].str.strip().str.replace("OOS=", "", regex=False)
    except Exception:
        hm["IS"]  = hm.get("is", "").astype(str)
        hm["OOS"] = hm.get("oos", "").astype(str)

    hm["Mode"] = hm["mode"].astype(str)
    metric_col = metric if metric in hm.columns else "Sharpe"
    hm[metric_col] = pd.to_numeric(hm[metric_col], errors="coerce")

    base = (
        alt.Chart(hm)
        .mark_rect()
        .encode(
            x=alt.X("IS:N",  title=f"IS ({is_unit_en})"),
            y=alt.Y("OOS:N", title=f"OOS ({oos_unit_en})"),
            color=alt.Color(f"{metric_col}:Q", title=f"{metric_col}",
                            scale=alt.Scale(scheme="viridis")),
        )
        .properties(height=320)
    )

    modes_unique = sorted(hm["Mode"].dropna().unique().tolist())
    if len(modes_unique) > 1:
        ch = base.facet(column=alt.Column("Mode:N", title="Mode"))
    else:
        ch = base.properties(title=f"Mode: {modes_unique[0] if modes_unique else 'n/a'}")

    st.altair_chart(ch, use_container_width=True)

# ----------------------------
# TAB 3 — Metrics distribution
# ----------------------------
with tab_metrics:
    st.subheader("📊 Metric distribution across the bundle")

    metric_col = metric if metric in df_bundle.columns else "Sharpe"
    vals = pd.to_numeric(df_bundle[metric_col], errors="coerce").dropna()

    if vals.empty:
        st.info("No values available for the selected metric.")
    else:
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("p5", f"{p5:.3f}")
        c2.metric("p25", f"{p25:.3f}")
        c3.metric("median", f"{p50:.3f}")
        c4.metric("p75", f"{p75:.3f}")
        c5.metric("p95", f"{p95:.3f}")

        hist = (
            alt.Chart(pd.DataFrame({metric_col: vals}))
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X(f"{metric_col}:Q", bin=alt.Bin(maxbins=45)),
                y=alt.Y("count()"),
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)

    st.caption("A tight distribution shifted to the right suggests a robust bundle.")

# ----------------------------
# TAB 4 — Downloads (side-by-side)
# ----------------------------
with tab_downloads:
    st.subheader("⬇️ Download")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download configurations table (CSV)",
            data=df_bundle.to_csv(index=False),
            file_name="wfb_bundle_summary.csv",
            mime="text/csv"
        )

    with c2:
        out = pd.DataFrame({k: v for k, v in oos_concat_map.items()})
        st.download_button(
            "Download concatenated OOS (all configs, CSV)",
            data=out.to_csv(),
            file_name="wfb_oos_concat_all.csv",
            mime="text/csv"
        )

    st.caption("Wide format: columns = configurations, rows = timestamps.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("© Walk-Forward Bundle Validator — built for clarity & robustness.")
