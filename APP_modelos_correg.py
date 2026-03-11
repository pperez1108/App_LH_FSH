# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:24:57 2026

@author: pperez
"""
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURACIÓN VISUAL CORPORATIVA ---
st.set_page_config(page_title="IA Dosificación", page_icon="💉", layout="centered")

st.markdown(f"""
    <style>
    .stApp {{
        background-color: #182430;
        color: #D2C2B0;
    }}
    .stButton>button {{
        background-color: #A78C6F !important;
        color: #182430 !important;
        border: none;
        font-weight: bold;
    }}
    .stMetric {{
        background-color: #3d4752;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #A78C6F;
    }}
    div[data-testid="stMetricValue"] {{
        color: #D2C2B0 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #676767 !important;
    }}
    header, footer {{
        visibility: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ACCESO (LOG IN) ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = None

    if st.session_state["password_correct"] is True:
        return True

    st.title("Acceso Médico Restringido 🔐")
    
    with st.container():
        user = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        
        if st.button("Entrar"):
            # Verificación contra Secrets
            if user == st.secrets["credentials"]["username"] and \
               password == st.secrets["credentials"]["password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.session_state["password_correct"] = False
                st.error("😕 Credenciales incorrectas. Por favor, inténtelo de nuevo.")
    return False

# --- 3. APLICACIÓN PRINCIPAL ---
if check_password():

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

    st.title("Predicción Dosis FSH y LH 💉")
    st.write("Introduzca los datos de la paciente para calcular la dosificación basada en el histórico clínico (2022-2026).")
    
    st.divider()

    # Formulario centralizado (sin sidebar)
    col_a, col_b = st.columns(2)
    
    with col_a:
        peso = st.number_input("Peso (kg)", value=67.0, step=0.1)
        altura = st.number_input("Altura (m)", value=1.68, step=0.01)
        edad = st.number_input("Edad", value=35, step=1)
    
    with col_b:
        amh = st.number_input("AMH (ng/ml)", value=0.38, step=0.01)
        rfa = st.number_input("RFA (folículos)", value=6, step=1)
        st.write("") # Espaciador
        calcular = st.button("CALCULAR DOSIS", use_container_width=True)

    if calcular:
        imc = peso / (altura ** 2)
        
        # Preparación de datos
        input_data = pd.DataFrame([[peso, altura, edad, amh, rfa, imc]], columns=features)
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
        
        # Predicciones
        fsh_pred = m_fsh1.predict(input_scaled)[0]
        fsh_final = round(fsh_pred / 25) * 25
        
        lh_class_val = float(m_lh1_class.predict(input_scaled)[0])
        lh_conf = max(m_lh1_class.predict_proba(input_scaled)[0])
        lh_reg_val = m_lh2_reg.predict(input_scaled)[0]
        lh_reg_final = round(lh_reg_val / 12.5) * 12.5
        
        # Resultados
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("FSH Sugerida", f"{fsh_final} UI")
            st.caption(f"Valor calculado: {fsh_pred:.1f}")

        with res_col2:
            if lh_conf >= 0.90 or lh_class_val == lh_reg_final:
                st.metric("LH Sugerida", f"{lh_class_val} UI")
            else:
                rango = sorted([lh_class_val, lh_reg_final])
                st.metric("Rango LH Sugerido", f"{rango[0]} - {rango[1]} UI")
            
            st.info(f"Confianza del modelo: {lh_conf:.1%}")

        st.divider()
        st.write(f"**Análisis:** Paciente con IMC de {imc:.1f}.")
        st.caption("⚠️ Nota: Esta herramienta es un soporte diagnóstico basado en datos históricos. El criterio final corresponde al médico.")

