# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:24:57 2026

@author: pperez
"""
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURACIÓN VISUAL Y LIMPIEZA DE INTERFAZ ---
st.set_page_config(page_title="IA Dosificación", page_icon="💉", layout="centered")

st.markdown(f"""
    <style>
    /* Ocultar Menú Superior, Botón de GitHub/Fork y Footer */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .viewerBadge_container__1QSob {{display: none !important;}}
    
    /* Fondo y texto general corporativo */
    .stApp {{
        background-color: #182430;
        color: #D2C2B0;
    }}
    
    /* Tamaño de letra para variables de entrada */
    label {{
        font-size: 1.25rem !important;
        color: #D2C2B0 !important;
        font-weight: 500;
    }}
    
    /* Estilo del botón CALCULAR */
    .stButton>button {{
        background-color: #A78C6F !important;
        color: #182430 !important;
        border: none;
        font-weight: bold;
        font-size: 1.4rem !important;
        height: 3.5em !important;
        margin-top: 1.5em;
        border-radius: 8px;
    }}
    
    /* Cuadros de resultados (Métricas) */
    div[data-testid="stMetric"] {{
        background-color: #3d4752;
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid #A78C6F;
    }}
    div[data-testid="stMetricValue"] > div {{
        font-size: 3rem !important; /* Resultado muy grande para visibilidad clínica */
        color: #D2C2B0 !important;
    }}
    div[data-testid="stMetricLabel"] > div {{
        font-size: 1.2rem !important;
        color: #A78C6F !important;
        text-transform: uppercase;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE ACCESO (Login) ---
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
            # Verifica contra los secretos de Streamlit Cloud
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
    st.markdown("#### Herramienta de soporte clínico (Basada en 559 punciones 2022-2026)")
    st.divider()

    # --- FILAS DE VARIABLES ---
    # Fila 1: Peso y Altura
    col1, col2 = st.columns(2)
    with col1:
        peso = st.number_input("Peso (kg)", value=67.0, step=0.1, format="%.2f")
    with col2:
        altura = st.number_input("Altura (m)", value=1.68, step=0.01, format="%.2f")

    # Fila 2: AMH y RFA
    col3, col4 = st.columns(2)
    with col3:
        amh = st.number_input("AMH (ng/ml)", value=0.38, step=0.01, format="%.2f")
    with col4:
        rfa = st.number_input("RFA (folículos)", value=6, step=1)

    # Fila 3: Edad sola
    edad = st.number_input("Edad (años)", value=35, step=1)

    # Fila 4: Botón abajo
    calcular = st.button("CALCULAR DOSIS PERSONALIZADA", use_container_width=True)

    if calcular:
        # Cálculo de IMC e Input
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
        
        # --- RESULTADOS VISUALES ---
        st.divider()
        res_c1, res_c2 = st.columns(2)
        
        with res_c1:
            st.metric("FSH SUGERIDA", f"{fsh_final} UI")
            st.caption(f"Tendencia matemática: {fsh_pred:.1f} UI")

        with res_c2:
            if lh_conf >= 0.90 or lh_class_val == lh_reg_final:
                st.metric("LH SUGERIDA", f"{lh_class_val} UI")
            else:
                rango = sorted([lh_class_val, lh_reg_final])
                st.metric("RANGO LH", f"{rango[0]} - {rango[1]} UI")
            
            st.info(f"Confianza: {lh_conf:.1%}")

        st.divider()
        st.write(f"**Paciente:** {edad} años | **IMC:** {imc:.1f}")
        st.caption("Esta herramienta es un soporte diagnóstico. El criterio clínico del médico es soberano.")


