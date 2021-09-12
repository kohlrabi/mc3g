import streamlit as st
import numpy as np
from mc3g import plot

st.set_page_config(page_title="2G/3G Monte Carlo", initial_sidebar_state="expanded")
st.title("2G/3G Monte Carlo")

with st.sidebar.container():
    N = st.sidebar.number_input("Anzahl Personen im Innenraum:", 0, 10_000, 100, 1, format='%d')
    p = st.sidebar.number_input("Prävalenz (%) (Standard: 1.0%):", 1e-3, 100., 1., 1e-4, format='%.2f')
    ve = st.sidebar.number_input("Impfstoffwirksamkeit (%) (Standard 66%):", -100, 100, 66, 1, format='%d')
    vr = st.sidebar.number_input("Impfquote (%) (Standard: 80%):", 0, 100, 80, 1, format='%d')
    sens = st.sidebar.number_input("Test-Sensitivität (%) (Standard: 80%, Minimum: 1%):", 1, 100, 80, 1, format='%d')
    spec = st.sidebar.number_input("Test-Spezifität (%) (Standard: 97%, Minimum: 1%):", 1, 100, 97, 1, format='%d')

fig = plot(vr/100, ve/100, p/100., N, sens/100, spec/100)
st.pyplot(fig)
