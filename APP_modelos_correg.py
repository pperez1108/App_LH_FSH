# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:24:57 2026

@author: pperez
"""
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURACIÓN VISUAL Y ESTILOS PERSONALIZADOS ---
st.set_page_config(page_title="IA Dosificación", page_icon="💉", layout="centered")

st.markdown(f"""
    <style>
    /* Fondo y texto general */
    .stApp {{
        background-color: #182430;
        color: #D2C2B0;
    }}
    /* Aumentar tamaño de etiquetas de entrada */
    label {{
        font-size: 1.2rem !important;
        color: #D2C2B0 !important;
    }}
    /* Estilo de los inputs */
    .stNumberInput input {{
        font-size: 1.1rem !important;
    }}
    /* Botón principal */
    .stButton>button {{
        background-color: #A78C6F !important;
        color: #182430 !important;
        border: none;
        font-weight: bold;
        font-size: 1.3rem !important;
        height: 3em !important;
        margin-top: 1em;
    }}
    /* Cuadros de métricas de resultados */
    div[data-testid="stMetric"] {{
        background-color: #3d4752;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #A78C6F;
    }}
    div[data-testid="stMetricValue"] > div {{
        font-size: 2.5rem !important; /* Resultado más grande */
        color: #D2C2B0 !important;
    }}
    div[data-testid="stMetricLabel"] > div {{
        font-size: 1.1rem !important;
        color: #A78C6F !important;
    }}
    /* Quitar menús por defecto */
    header, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ACCESO ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = None

    if st.session_state["password_correct"] is True:
        return True

    st.title("Acceso Médico Restringido 🔐")
    with st.container():
        user = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        if st.button("ENTRAR"):
            if user == st.secrets["credentials"]["username"] and \
               password == st.secrets["credentials"]["password"]:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.session_state["password_correct"] = False
                st.error("Credenciales incorrectas.")
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
    st.markdown("##### Introduzca los datos clínicos para la estimulación")
    st.divider()

    # --- FILAS DE INTRODUCCIÓN DE VARIABLES ---
    # Fila 1: Peso y Altura
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        peso = st.number_input("Peso (kg)", value=67.0, step=0.1, format="%.2f")
    with row1_col2:
        altura = st.number_input("Altura (m)", value=1.68, step=0.01, format="%.2f")

    # Fila 2: AMH y RFA
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        amh = st.number_input("AMH (ng/ml)", value=0.38, step=0.01, format="%.2f")
    with row2_col2:
        rfa = st.number_input("RFA (folículos)", value=6, step=1)

    # Fila 3: Edad (Sola en su fila)
    edad = st.number_input("Edad (años)", value=35, step=1)

    # Fila 4: Botón (Solo abajo)
    calcular = st.button("CALCULAR DOSIS PERSONALIZADA", use_container_width=True)

    if calcular:
        imc = peso / (altura ** 2)
        input_data = pd.DataFrame([[peso, altura, edad, amh, rfa, imc]], columns=features)
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
        
        # Predicciones
        fsh_pred = m_fsh1.predict(input_scaled)[0]
        fsh_final = round(fsh_pred / 25) * 25
        
        lh_class_val = float(m_lh1_class.predict(input_scaled)[0])
        lh_conf = max(m_lh1_class.predict_proba(input_scaled)[0])
        lh_reg_val = m_lh2_reg.predict(input_scaled)[0]
        lh_reg_final = round(lh_reg_val / 12.5) * 12.5
        
        # --- MOSTRAR RESULTADOS ---
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("FSH SUGERIDA", f"{fsh_final} UI")
            st.caption(f"Tendencia calculada: {fsh_pred:.1f} UI")

        with res_col2:
            if lh_conf >= 0.90 or lh_class_val == lh_reg_final:
                st.metric("LH SUGERIDA", f"{lh_class_val} UI")
            else:
                rango = sorted([lh_class_val, lh_reg_final])
                st.metric("RANGO LH SUGERIDO", f"{rango[0]} - {rango[1]} UI")
            
            st.info(f"Nivel de confianza: {lh_conf:.1%}")

        st.divider()
        st.write(f"**Análisis Clínico:** IMC calculado de {imc:.1f}. Basado en histórico 2022-2026.")
        st.caption("Esta herramienta es un soporte diagnóstico. El criterio médico final prevalece.")


