import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    # Plot cumulative returns
    if "returns" in data.columns:
        data["cumulative"] = (1 + data["returns"]).cumprod()

        fig, ax = plt.subplots()
        ax.plot(data["cumulative"], label="Cumulative Return")
        ax.set_title("Equity Line")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("Il file deve contenere una colonna 'returns'.")

st.success(
    "✨ Consiglio: per disattivare l'auto-reload ed evitare errori con i watcher, aggiungi nel file `.streamlit/config.toml` la riga `fileWatcherType = \"none\"` sotto la sezione `[server]`."
)
