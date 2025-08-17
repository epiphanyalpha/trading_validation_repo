import streamlit as st
import pandas as pd
import numpy as np

# Usare Altair per i grafici (già incluso in Streamlit)
import altair as alt

st.set_page_config(page_title="Dashboard Validazione Trading", layout="wide")

st.title("📊 Dashboard Validazione Trading")

st.markdown(
    """
    Questa app Streamlit mostra:

    - **Test di overfitting** (López de Prado, CSCV/PBO)
    - **Walk-Forward Bundle**

    ---
    """
)

st.info("L'app è stata avviata correttamente ✅. Se vedi questa schermata, la configurazione è a posto.")

# File uploader
uploaded_file = st.file_uploader("📂 Carica un file CSV con i risultati della strategia (colonne: date, returns)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Anteprima dati")
    st.write(data.head())

    # Plot cumulative returns con Altair
    if "returns" in data.columns:
        data["cumulative"] = (1 + data["returns"]).cumprod()

        chart = (
            alt.Chart(data.reset_index())
            .mark_line()
            .encode(
                x=alt.X("index", title="Periodo"),
                y=alt.Y("cumulative", title="Cumulative Return"),
                tooltip=["index", "cumulative"]
            )
            .properties(title="Equity Line", width=700, height=400)
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.error("Il file deve contenere una colonna 'returns'.")

st.success(
    "✨ Consiglio: per disattivare l'auto-reload ed evitare errori con i watcher, aggiungi nel file `.streamlit/config.toml` la riga `fileWatcherType = \"none\"` sotto la sezione `[server]`."
)
