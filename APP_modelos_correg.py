# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:24:57 2026

@author: pperez
"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IA Dosificación", page_icon="💉")

@st.cache_resource
def load_assets():
    return (
        joblib.load('scaler.pkl'), 
        joblib.load('model_fsh1.pkl'), 
        joblib.load('model_lh1.pkl'), 
        joblib.load('model_lh2.pkl'),
        joblib.load('features.pkl')
    )

scaler, m_fsh1, m_lh1_class, m_lh2_reg, features = load_assets()

st.title("Predicción Dosis FSH y LH en estimulación ovárica 💉")

with st.sidebar:
    st.header("Datos Paciente")
    # Valores por defecto basados en tus datos de ejemplo
    peso = st.number_input("Peso (kg)", value=67.0)
    altura = st.number_input("Altura (m)", value=1.68)
    edad = st.number_input("Edad", value=35)
    amh = st.number_input("AMH", value=0.38)
    rfa = st.number_input("RFA", value=6)
    calcular = st.button("Calcular Dosis", type="primary")

if calcular:
    imc = peso / (altura ** 2)
    
    # Crear DataFrame de entrada
    input_data = pd.DataFrame([[peso, altura, edad, amh, rfa, imc]], columns=features)
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
    
    # 1. Predicción FSH
    fsh_pred = m_fsh1.predict(input_scaled)[0]
    fsh_final = round(fsh_pred / 25) * 25
    
    # 2. Predicción LH Clasificación (Modelo 1)
    lh_class_val = float(m_lh1_class.predict(input_scaled)[0])
    lh_conf = max(m_lh1_class.predict_proba(input_scaled)[0])
    
    # 3. Predicción LH Regresión (Modelo 2)
    lh_reg_val = m_lh2_reg.predict(input_scaled)[0]
    lh_reg_final = round(lh_reg_val / 12.5) * 12.5
    
    # --- MOSTRAR RESULTADOS ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("FSH Sugerida", f"{fsh_final} UI")
        st.caption(f"Valor bruto calculado: {fsh_pred:.1f}")

    with col2:
        # Lógica de visualización corregida
        if lh_conf >= 0.90 or lh_class_val == lh_reg_final:
            # Si hay alta confianza O ambos modelos coinciden, mostramos valor único
            st.metric("LH Sugerida", f"{lh_class_val} UI")
        else:
            # Si hay duda y los valores son distintos, mostramos rango
            rango = sorted([lh_class_val, lh_reg_final])
            st.metric("Rango LH Sugerido", f"{rango[0]} - {rango[1]} UI")
        
        # Mensaje de confianza unificado sin etiquetas de texto
        st.info(f"Confianza: {lh_conf:.1%}  ")

    st.divider()
    st.write(f"**Análisis de consistencia:** El IMC calculado es {imc:.1f}. " 
             f"El modelo de clasificación ha detectado una probabilidad dominante para la dosis de {lh_class_val} UI.")
