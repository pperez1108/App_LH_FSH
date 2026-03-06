# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:50:58 2026

@author: pperez
"""

import streamlit as st
import joblib
import numpy as np

# -------------------------
# Cargar modelo
# -------------------------

model = joblib.load("modelo_dosisFSH_LH.pkl")

st.set_page_config(page_title="Predicción Dosis Estimulación", layout="centered")

st.title("💊 Predicción de Dosis en Estimulación Ovárica")

st.markdown("Introduce los datos clínicos de la paciente:")

# -------------------------
# Inputs del usuario
# -------------------------

edad = st.number_input("Edad", 18, 55, 35)
amh = st.number_input("AMH", 0.0, 20.0, 2.0)
fsh = st.number_input("FSH", 0.0, 35.0, 6.0)
lh = st.number_input("LH", 0.0, 35.0, 5.0)
rfa = st.number_input("RFA", 0, 100, 10)
imc = st.number_input("IMC", 15.0, 40.0, 22.0)

# -------------------------
# Predicción
# -------------------------

if st.button("Calcular dosis recomendada"):

    input_data = np.array([[edad, amh, fsh, lh, rfa, imc]])
    prediction = model.predict(input_data)

    dosis_lh = round(prediction[0][0], 2)
    dosis_fsh = round(prediction[0][1], 2)

    st.success("### Resultados recomendados")

    col1, col2 = st.columns(2)

    col1.metric("Dosis FSH", f"{dosis_lh} UI")
    col2.metric("Dosis LH", f"{dosis_fsh} UI")