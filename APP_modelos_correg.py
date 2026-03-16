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
        font-size: 1.25rem !important;
        color: #D2C2B0 !important;
        font-weight: 500;
    }}
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
    div[data-testid="stMetric"] {{
        background-color: #3d4752;
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid #A78C6F;
    }}
    div[data-testid="stMetricValue"] > div {{
        font-size: 3rem !important;
        color: #D2C2B0 !important;
    }}
    div[data-testid="stMetricLabel"] > div {{
        font-size: 1.2rem !important;
        color: #A78C6F !important;
        text-transform: uppercase;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. DICCIONARIO DE MEDICAMENTOS ---
MEDICAMENTOS = {
    "GONAL": (1.0, 0.0),
    "PUREGON": (1.0, 0.0),
    "BEMFOLA": (1.0, 0.0),
    "REKOVELLE": (1.0, 0.0),
    "FOSTIPUR": (1.0, 0.0),
    "OVELAP": (1.0, 0.0),
    "ELONVA": (1.0, 0.0),
    "PERGOVERIS": (1.0, 0.5), # 150 FSH / 75 LH
    "MERIOFERT": (1.0, 1.0),  # 75 FSH / 75 LH
    "MENOPUR": (0.0, 1.0),
    "LEPORI": (0.0, 1.0),
    "NINGUNO": (0.0, 0.0)
}

# --- 3. GESTIÓN DE ACCESO ---
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

    st.title("Predicción Dosis FSH y LH 💉")
    st.markdown("#### Herramienta de soporte clínico (Basada en 559 punciones 2022-2026)")
    st.divider()

    # --- ENTRADA DE DATOS ---
    col1, col2 = st.columns(2)
    with col1:
        peso = st.number_input("Peso (kg)", value=67.0, step=0.1, format="%.2f")
        amh = st.number_input("AMH (ng/ml)", value=0.38, step=0.01, format="%.2f")
    with col2:
        altura = st.number_input("Altura (m)", value=1.68, step=0.01, format="%.2f")
        rfa = st.number_input("RFA (folículos)", value=6, step=1)
    
    edad = st.number_input("Edad (años)", value=35, step=1)
    
    calcular = st.button("CALCULAR DOSIS PERSONALIZADA", use_container_width=True)

    # Inicializamos variables en session_state para que la calculadora manual las use
    if "fsh_manual" not in st.session_state: st.session_state.fsh_manual = 150.0
    if "lh_manual" not in st.session_state: st.session_state.lh_manual = 75.0

    if calcular:
        imc = peso / (altura ** 2)
        input_data = pd.DataFrame([[peso, altura, edad, amh, rfa, imc]], columns=features)
        input_scaled = pd.DataFrame(scaler.transform(input_data), columns=features)
        
        # Predicciones FSH
        fsh_pred = m_fsh1.predict(input_scaled)[0]
        fsh_final = round(fsh_pred / 25) * 25
        st.session_state.fsh_manual = float(fsh_final)
        
        # Predicciones LH (Tu lógica original de rangos)
        lh_class_val = float(m_lh1_class.predict(input_scaled)[0])
        lh_conf = max(m_lh1_class.predict_proba(input_scaled)[0])
        lh_reg_val = m_lh2_reg.predict(input_scaled)[0]
        lh_reg_final = round(lh_reg_val / 12.5) * 12.5
        
        st.session_state.lh_manual = float(lh_class_val)

        # --- RESULTADOS VISUALES IA ---
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

        st.write(f"**Paciente:** {edad} años | **IMC:** {imc:.1f}")

    # --- 5. CALCULADORA DE MEDICACIÓN MANUAL (Ajuste Médico) ---
    st.divider()
    st.subheader("Configuración de Medicación Comercial 💊")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        fsh_target = st.number_input("Ajuste FSH (UI)", value=st.session_state.fsh_manual, step=12.5)
        med_fsh = st.selectbox("Fármaco base FSH", ["GONAL", "PUREGON", "BEMFOLA", "REKOVELLE", "FOSTIPUR", "PERGOVERIS", "MERIOFERT"])

    with col_m2:
        lh_target = st.number_input("Ajuste LH (UI)", value=st.session_state.lh_manual, step=12.5)
        med_lh = st.selectbox("Fármaco base LH / Segundo", ["MENOPUR", "LEPORI", "PERGOVERIS", "MERIOFERT", "NINGUNO"])

    if st.button("CALCULAR VIALES / UNIDADES"):
        fsh_a, lh_a = MEDICAMENTOS[med_fsh]
        fsh_b, lh_b = MEDICAMENTOS[med_lh] if med_lh != "NINGUNO" else (0.0, 0.0)

        det = (fsh_a * lh_b) - (fsh_b * lh_a)
        
        if abs(det) < 0.001:
            # Si los medicamentos son del mismo tipo (ej. ambos solo FSH)
            qty_a = fsh_target / fsh_a if fsh_a > 0 else 0
            qty_b = 0
            if med_lh != "NINGUNO": st.warning("Combinación redundante. Se prioriza el primer fármaco.")
        else:
            # Sistema de ecuaciones para encontrar la dosis exacta de cada uno
            qty_a = (fsh_target * lh_b - lh_target * fsh_b) / det
            qty_b = (fsh_a * lh_target - lh_a * fsh_target) / det

        # Aplicamos redondeo a 12.5 para ser realistas con las plumas
        qty_a_final = max(0.0, round(qty_a / 12.5) * 12.5)
        qty_b_final = max(0.0, round(qty_b / 12.5) * 12.5)

        st.markdown("### Dosis a administrar:")
        c_res1, c_res2 = st.columns(2)
        c_res1.success(f"**{med_fsh}**: {qty_a_final} UI")
        if med_lh != "NINGUNO":
            c_res2.success(f"**{med_lh}**: {qty_b_final} UI")
        
        # Verificación final de lo que recibe el paciente tras el redondeo
        total_fsh = (qty_a_final * fsh_a) + (qty_b_final * fsh_b)
        total_lh = (qty_a_final * lh_a) + (qty_b_final * lh_b)
        st.caption(f"Dosis final real con redondeo: FSH {total_fsh} | LH {total_lh}")

    st.divider()
    st.caption("Esta herramienta es un soporte diagnóstico. El criterio clínico del médico es soberano.")

