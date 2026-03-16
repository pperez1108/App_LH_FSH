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
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stApp {{
        background-color: #182430;
        color: #D2C2B0;
    }}
    label {{
        font-size: 1.1rem !important;
        color: #D2C2B0 !important;
    }}
    .stButton>button {{
        background-color: #A78C6F !important;
        color: #182430 !important;
        font-weight: bold;
        border-radius: 8px;
    }}
    div[data-testid="stMetric"] {{
        background-color: #3d4752;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #A78C6F;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DICCIONARIO DE MEDICAMENTOS (Basado en tu tabla) ---
# Formato: "Nombre": (Factor_FSH, Factor_LH)
MEDICAMENTOS = {
    "GONAL": (1.0, 0.0),
    "PUREGON": (1.0, 0.0),
    "BEMFOLA": (1.0, 0.0),
    "REKOVELLE": (1.0, 0.0),
    "FOSTIPUR": (1.0, 0.0),
    "OVELAP": (1.0, 0.0),
    "ELONVA": (1.0, 0.0),
    "PERGOVERIS": (1.0, 0.5), # Ratio 2:1 (150 FSH / 75 LH)
    "MERIOFERT": (1.0, 1.0),  # Ratio 1:1 (75 FSH / 75 LH)
    "MENOPUR": (0.0, 1.0),    # Actividad LH predominante
    "LEPORI": (0.0, 1.0),
    "NINGUNO": (0.0, 0.0)
}

# --- 3. GESTIÓN DE ACCESO (Login) ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = None
    if st.session_state["password_correct"] is True:
        return True
    st.title("Acceso Médico Restringido 🔐")
    user = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("ENTRAR"):
        if user == st.secrets["credentials"]["username"] and \
           password == st.secrets["credentials"]["password"]:
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("Credenciales incorrectas.")
    return False

# --- 4. APLICACIÓN PRINCIPAL ---
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

    st.title("IA Dosificación & Calculadora 💉")
    
    # --- BLOQUE 1: PREDICCIÓN IA ---
    with st.expander("1. Predicción del Modelo (IA)", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            peso = st.number_input("Peso (kg)", value=67.0, step=0.1)
            amh = st.number_input("AMH (ng/ml)", value=0.38, step=0.01)
        with col2:
            altura = st.number_input("Altura (m)", value=1.68, step=0.01)
            rfa = st.number_input("RFA (folículos)", value=6, step=1)
        with col3:
            edad = st.number_input("Edad (años)", value=35, step=1)
        
        calcular_ia = st.button("PREDECIR CON IA", use_container_width=True)

    fsh_final = 150.0  # Valores por defecto para la calculadora manual
    lh_final = 75.0

    if calcular_ia:
        imc = peso / (altura ** 2)
        input_data = pd.DataFrame([[peso, altura, edad, amh, rfa, imc]], columns=features)
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
        
        fsh_pred = m_fsh1.predict(input_scaled)[0]
        fsh_final = float(round(fsh_pred / 25) * 25)
        
        lh_class_val = float(m_lh1_class.predict(input_scaled)[0])
        lh_conf = max(m_lh1_class.predict_proba(input_scaled)[0])
        lh_reg_val = m_lh2_reg.predict(input_scaled)[0]
        lh_final = float(round(lh_reg_val / 12.5) * 12.5)
        
        c1, c2 = st.columns(2)
        c1.metric("FSH SUGERIDA", f"{fsh_final} UI")
        c2.metric("LH SUGERIDA", f"{lh_final} UI")
        st.caption(f"Paciente: {edad} años | IMC: {imc:.1f}")

    # --- BLOQUE 2: CALCULADORA DE MEDICACIÓN (AJUSTE MÉDICO) ---
    st.divider()
    st.subheader("2. Ajuste de Medicación Comercial")
    st.markdown("Seleccione las dosis finales y los fármacos para calcular la distribución.")

    col_man1, col_man2 = st.columns(2)
    with col_man1:
        fsh_target = st.number_input("Dosis FSH Final (UI)", value=fsh_final, step=12.5)
        med_1 = st.selectbox("Medicamento 1 (Base)", list(MEDICAMENTOS.keys()), index=0)
    
    with col_man2:
        lh_target = st.number_input("Dosis LH Final (UI)", value=lh_final, step=12.5)
        med_2 = st.selectbox("Medicamento 2 (Ajuste)", list(MEDICAMENTOS.keys()), index=7) # Por defecto Pergoveris o Menopur

    if st.button("CALCULAR DISTRIBUCIÓN DE VIALES"):
        fsh_a, lh_a = MEDICAMENTOS[med_1]
        fsh_b, lh_b = MEDICAMENTOS[med_2]

        # Sistema de ecuaciones:
        # (qty_a * fsh_a) + (qty_b * fsh_b) = fsh_target
        # (qty_a * lh_a) + (qty_b * lh_b) = lh_target
        
        det = (fsh_a * lh_b) - (fsh_b * lh_a)

        if abs(det) < 0.001:
            # Caso donde los medicamentos son proporcionales o iguales
            st.warning("Combinación linealmente dependiente. El sistema usará el Medicamento 1 prioritariamente.")
            qty_a = fsh_target / fsh_a if fsh_a > 0 else 0
            qty_b = 0
        else:
            qty_a = (fsh_target * lh_b - lh_target * fsh_b) / det
            qty_b = (fsh_a * lh_target - lh_a * fsh_target) / det

        # Redondeo clínico (múltiplos de 12.5 para plumas)
        qty_a_final = max(0.0, round(qty_a / 12.5) * 12.5)
        qty_b_final = max(0.0, round(qty_b / 12.5) * 12.5)

        st.markdown("### Plan de Administración:")
        res1, res2 = st.columns(2)
        with res1:
            st.success(f"**{med_1}:** {qty_a_final} UI")
        with res2:
            if med_2 != "NINGUNO":
                st.success(f"**{med_2}:** {qty_b_final} UI")
        
        # Validación de dosis resultante
        fsh_real = (qty_a_final * fsh_a) + (qty_b_final * fsh_b)
        lh_real = (qty_a_final * lh_a) + (qty_b_final * lh_b)
        st.info(f"Dosis total alcanzada: FSH {fsh_real} UI | LH {lh_real} UI")

    st.divider()
    st.caption("Esta herramienta es un soporte diagnóstico. El criterio clínico del médico es soberano.")

