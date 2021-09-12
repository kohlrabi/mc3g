import streamlit as st
import numpy as np
from mc3g import plot

st.set_page_config(page_title="2G/3G Monte Carlo", initial_sidebar_state="expanded")
st.title("2G/3G Monte Carlo")

with st.sidebar.container():
    N = st.sidebar.slider("Anzahl Personen im Innenraum:", 0, 500, 200, 10)
    prevalence = np.logspace(-1, 1, 91, endpoint=True)
    p = st.sidebar.select_slider("Prävalenz (%) (Standard: 1.0%):", prevalence, 1, format_func=lambda x: f'{x:.2f}%')
    ve = st.sidebar.slider("Impfstoffwirksamkeit (%) (Standard 66%):", -100, 100, 66, 1, format='%d%%')
    vq = st.sidebar.slider("Impfquote (%) (Standard: 80%):", 0, 100, 80, 1, format='%d%%')
    sens = st.sidebar.slider("Test-Sensitivität (%) (Standard: 80%, Minimum: 1%):", 1, 100, 80, 1, format='%d%%')
    spec = st.sidebar.slider("Test-Spezifität (%) (Standard: 97%, Minimum: 1%):", 1, 100, 97, 1, format='%d%%')

fig = plot(vq/100, ve/100, p/100., N, sens/100, spec/100)
st.pyplot(fig)
