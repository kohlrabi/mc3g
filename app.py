import streamlit as st
import numpy as np
from mc3g_plot import plot

st.set_page_config(page_title="2G/3G Monte Carlo", initial_sidebar_state="expanded")
st.title("2G/3G Monte Carlo")

with st.sidebar.container():
    N = st.sidebar.number_input("Anzahl Personen auf dem Event (Minimum: 0, Maximum: 10.000, Standard: 100):", 0, 10_000, 100, 1, format='%d')
    p = st.sidebar.number_input("Prävalenz (%) (Minimum: 0.01%, Maximum: 100%, Standard: 1.0%):", 1e-2, 100., 1., 1e-3, format='%.2f')
    sens = st.sidebar.number_input("Test-Sensitivität (%) (Minimum: 1%, Maximum: 100%, Standard: 80%):", 1., 100., 80., 1e-2, format='%.2f')
    spec = st.sidebar.number_input("Test-Spezifität (%) (Minimum: 1%, Maximum: 100%, Standard: 97%):", 1., 100., 97., 1e-2, format='%.2f')
    ve = st.sidebar.number_input("Impfstoffwirksamkeit (%) (Minimum: -100%, Maximum: 100%, Standard 66%):", -100., 100., 80., 1e-2, format='%.2f')
    vr = st.sidebar.number_input("Impfquote (%) (Minimum: 0%, Maximum: 100%, Standard: 80%):", 0., 100., 80., 1e-2, format='%.2f')

fig = plot(vr/100, ve/100, p/100., N, sens/100, spec/100)
st.pyplot(fig)
