import streamlit as st
import numpy as np
from mc3g import plot

st.set_page_config(page_title="2G/3G Monte Carlo", initial_sidebar_state="expanded")
st.title("2G/3G Monte Carlo")

with st.sidebar.container():
    N = st.sidebar.slider("Anzahl Studenten im Hörsaal:", 0, 500, 200, 10)
    prevalence = np.logspace(-1, 1, 61, endpoint=True)
    p = st.sidebar.select_slider("Prävalenz (%):", prevalence, 1)
    ve = st.sidebar.slider("Impfstoffwirksamkeit:", 0., 100., 66., 1.)
    vq = st.sidebar.slider("Impfquote:", 0., 100., 80., 1.)

fig = plot(vq/100, ve/100, p/100., N)
st.pyplot(fig)
