# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:24:57 2026

@author: pperez
"""
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# --- 1. CONFIGURACIÓN VISUAL ---
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
        font-size: 1.2rem !important;
        color: #D2C2B0 !important;
        font-weight: 500;
    }}
    .stButton>button {{
        background-color: #A78C6F !important;
        color: #182430 !important;
        border: none;
        font-weight: bold;
        font-size: 1.3rem !important;
        border-radius: 8px;
    }}
    div[data-testid="stMetric"] {{
        background-color: #3d4752;
        padding: 20px;
        border-radius: 12px;
        border-left: 8px solid #A78C6F;
    }}
    /* Estilo de las cajas de dosis finales */
    .dose-box {{
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }}
    .fsh-box {{ background-color: rgba(76, 175, 80, 0.15); border: 2px solid #4CAF50; color: #81C784; }}
    .lh-box {{ background-color: rgba(33, 150, 243, 0.15); border: 2px solid #2196F3; color: #64B5F6; }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DICCIONARIO DE MEDICAMENTOS Y LISTAS FILTRADAS ---
MEDICAMENTOS = {
    "GONAL": (1.0, 0.0),
    "PUREGON": (1.0, 0.0),
    "BEMFOLA": (1.0, 0.0),
    "REKOVELLE": (1.0, 0.0),
    "FOSTIPUR": (1.0, 0.0),
    "OVELAP": (1.0, 0.0),
    "PERGOVERIS": (1.0, 0.5), 
    "MERIOFERT": (1.0, 1.0),  
    "MENOPUR": (0.0, 1.0),
    "NINGUNO": (0.0, 0.0)
}

# Listas específicas para los selectores
LISTA_FSH = ["GONAL", "PUREGON", "BEMFOLA", "REKOVELLE", "FOSTIPUR", "OVELAP", "PERGOVERIS", "MERIOFERT"]
LISTA_LH = ["MENOPUR", "PERGOVERIS", "MERIOFERT", "NINGUNO"]

# --- 3. GESTIÓN DE ACCESO ---
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
    st.divider()

    # --- ENTRADA DE VARIABLES ---
    col1, col2 = st.columns(2)
    with col1:
        peso = st.number_input("Peso (kg)", value=67.0, step=0.1)
        amh = st.number_input("AMH (ng/ml)", value=0.38, step=0.01)
    with col2:
        altura = st.number_input("Altura (m)", value=1.68, step=0.01)
        rfa = st.number_input("RFA (folículos)", value=6, step=1)
    
    edad = st.number_input("Edad (años)", value=35, step=1)
    
    # Session state para comunicación entre bloques
    if "fsh_m" not in st.session_state: st.session_state.fsh_m = 150.0
    if "lh_m" not in st.session_state: st.session_state.lh_m = 75.0

    if st.button("CALCULAR DOSIS PERSONALIZADA", use_container_width=True):
        imc = peso / (altura ** 2)
        input_data = pd.DataFrame([[peso, altura, edad, amh, rfa, imc]], columns=features)
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
        
        fsh_pred = m_fsh1.predict(input_scaled)[0]
        fsh_final = round(fsh_pred / 25) * 25
        st.session_state.fsh_m = float(fsh_final)
        
        lh_class_val = float(m_lh1_class.predict(input_scaled)[0])
        lh_conf = max(m_lh1_class.predict_proba(input_scaled)[0])
        lh_reg_val = m_lh2_reg.predict(input_scaled)[0]
        lh_reg_final = round(lh_reg_val / 12.5) * 12.5
        st.session_state.lh_m = float(lh_class_val)

        # Resultados IA (Original)
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

    # --- 5. CALCULADORA DE MEDICACIÓN (Ajuste por el médico) ---
    st.divider()
    st.subheader("Configuración de Medicación Real 💊")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fsh_target = st.number_input("Dosis FSH deseada (UI)", value=st.session_state.fsh_m, step=12.5)
        med_fsh = st.selectbox("Seleccionar fármaco base FSH", LISTA_FSH)

    with col_m2:
        lh_target = st.number_input("Dosis LH deseada (UI)", value=st.session_state.lh_m, step=12.5)
        med_lh = st.selectbox("Seleccionar fármaco base LH", LISTA_LH)

    if st.button("CALCULAR VIALES / UNIDADES"):
        f_a, l_a = MEDICAMENTOS[med_fsh]
        f_b, l_b = MEDICAMENTOS[med_lh] if med_lh != "NINGUNO" else (0.0, 0.0)

        # Resolución del sistema de ecuaciones
        det = (f_a * l_b) - (f_b * l_a)
        if abs(det) < 0.001:
            qty_a = fsh_target / f_a if f_a > 0 else 0
            qty_b = 0
            if med_lh != "NINGUNO": st.warning("Combinación con el mismo ratio. Usando fármaco base.")
        else:
            qty_a = (fsh_target * l_b - lh_target * f_b) / det
            qty_b = (f_a * lh_target - l_a * fsh_target) / det

        st.markdown("### Dosis a administrar:")
        
        def display_dose(name, val, color_class):
            if name == "REKOVELLE":
                mcg = val / 12.5
                text = f"{name}: {mcg:.2f} µg"
            else:
                ui = max(0.0, round(val / 12.5) * 12.5)
                text = f"{name}: {ui} UI"
            st.markdown(f'<div class="dose-box {color_class}">{text}</div>', unsafe_allow_html=True)

        res_p1, res_p2 = st.columns(2)
        with res_p1:
            display_dose(med_fsh, qty_a, "fsh-box")
        with res_p2:
            if med_lh != "NINGUNO":
                display_dose(med_lh, qty_b, "lh-box")

        total_fsh = (qty_a * f_a) + (qty_b * f_b)
        total_lh = (qty_a * l_a) + (qty_b * l_b)
        st.caption(f"Verificación final: FSH {total_fsh:.1f} UI | LH {total_lh:.1f} UI")

    st.divider()
    st.caption("Esta herramienta es un soporte diagnóstico. El criterio clínico del médico es soberano.")
