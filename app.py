import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import plotly.graph_objects as go
import plotly.express as px
import json
import requests
from PIL import Image
import io
import base64
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------- CONFIG PAGE ----------------
st.set_page_config(
    page_title="❤️ CardioRisk AI - Prédiction Intelligente",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- FONCTIONS UTILITAIRES ----------------
def load_heart_image():
    """Charge l'image de cœur PNG depuis le chemin local du projet (chemin relatif)."""
    try:
        # Chemin relatif (dans le même dossier que app.py)
        heart_path = os.path.join(os.path.dirname(__file__), "heart.png")

        if os.path.exists(heart_path):
            heart_img = Image.open(heart_path)
            return heart_img
        else:
            st.warning(f"⚠️ Image de cœur non trouvée à : {heart_path}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'image : {str(e)}")
        return None


def get_base64_image(img):
    """Convertit une image PIL en base64."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# ---------------- CSS CUSTOM AVANCÉ AVEC EFFETS DE RISQUE ÉLEVÉ ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #07111f;
    --panel: #101e31;
    --panel-2: #14263d;
    --line: rgba(148, 163, 184, .16);
    --text: #edf5ff;
    --muted: #91a4bb;
    --cyan: #39d5e8;
    --red: #ff4d5e;
    --green: #20c997;
    --amber: #ffb454;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    color: var(--text);
    background:
        radial-gradient(circle at 12% 0%, rgba(57,213,232,.10), transparent 28%),
        radial-gradient(circle at 92% 12%, rgba(255,77,94,.10), transparent 25%),
        linear-gradient(145deg, #07111f 0%, #0b1728 52%, #091321 100%);
}
[data-testid="stHeader"] { background: rgba(7,17,31,.72); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a2b 0%, #101f33 100%);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * { color: var(--text); }
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -.035em;
    color: var(--text);
}
p, label, .stMarkdown { color: #dce8f6; }
small, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* Cœur rouge : conserver l'image originale et son battement */
.heart-beat-png {
    display: block;
    margin: 0 auto;
    object-fit: contain;
    animation: heartbeat 1.35s ease-in-out infinite;
    filter: drop-shadow(0 0 14px rgba(255,65,82,.55))
            drop-shadow(0 0 34px rgba(255,65,82,.20));
    transform-origin: center;
}
@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    12% { transform: scale(1.075); }
    24% { transform: scale(1); }
    36% { transform: scale(1.12); }
    52% { transform: scale(1); }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
}

/* En-tête */
.header-container { margin: .5rem 0 1.5rem; }
.title-glitch {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(2.2rem, 5vw, 4rem);
    line-height: 1.05;
    font-weight: 800;
    text-align: center;
    letter-spacing: .04em;
    background: linear-gradient(90deg, #f5fbff 5%, #55ddeb 48%, #ff6474 95%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 8px 35px rgba(57,213,232,.12);
    margin-bottom: .65rem;
}
.hero-subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 1.05rem;
}
div[data-testid="stMarkdownContainer"] hr {
    border-color: var(--line);
}

/* Panneaux et cartes */
.cyber-card {
    background: linear-gradient(145deg, rgba(19,36,57,.94), rgba(12,26,43,.94)) !important;
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.35rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 18px 50px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.035);
    position: relative;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16,30,49,.68);
    border: 1px solid var(--line);
    border-radius: 20px;
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(20,38,61,.95), rgba(12,27,45,.95));
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 16px 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,.14);
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600; }
[data-testid="stMetricValue"] { color: #f5fbff !important; font-family: 'Space Grotesk', sans-serif; }
[data-testid="stMetricDelta"] { font-weight: 700; }

/* Contrôles */
.stButton > button, .stDownloadButton > button {
    border: 1px solid rgba(57,213,232,.35);
    border-radius: 13px;
    color: #061522;
    background: linear-gradient(100deg, #48d9e8, #78e4d2);
    font-weight: 800;
    padding: .72rem 1.1rem;
    transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
    box-shadow: 0 8px 24px rgba(57,213,232,.14);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px);
    filter: brightness(1.06);
    box-shadow: 0 12px 30px rgba(57,213,232,.24);
    border-color: var(--cyan);
}
.stButton > button:focus, .stDownloadButton > button:focus {
    box-shadow: 0 0 0 3px rgba(57,213,232,.22);
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--cyan);
    border: 2px solid #e8fbff;
    box-shadow: 0 0 0 4px rgba(57,213,232,.13);
}
.stSlider [data-testid="stTickBar"] { background: rgba(57,213,232,.35); }
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div {
    background: #10243a !important;
    color: var(--text) !important;
    border-color: rgba(148,163,184,.22) !important;
    border-radius: 11px !important;
}
[data-baseweb="tab-list"] {
    gap: .45rem;
    background: rgba(7,17,31,.55);
    border: 1px solid var(--line);
    padding: .4rem;
    border-radius: 14px;
}
[data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: var(--muted);
    padding: .65rem 1rem;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #071522 !important;
    background: linear-gradient(100deg, #48d9e8, #78e4d2);
    font-weight: 800;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--cyan), var(--green));
}
[data-testid="stAlert"] { border-radius: 14px; }

/* Niveaux de risque : code couleur cohérent */
.risk-badge {
    display: inline-block;
    padding: 11px 24px;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: .08em;
    margin: 12px 0;
    border: 1px solid rgba(255,255,255,.15);
}
.risk-low {
    color: #04251c;
    background: linear-gradient(100deg, #20c997, #7be4bd);
    box-shadow: 0 8px 26px rgba(32,201,151,.2);
}
.risk-medium {
    color: #392307;
    background: linear-gradient(100deg, #ffb454, #ffd17c);
    box-shadow: 0 8px 26px rgba(255,180,84,.2);
}
.risk-high {
    color: white;
    background: linear-gradient(100deg, #ff4d5e, #ff7a69);
    box-shadow: 0 8px 26px rgba(255,77,94,.25);
    animation: riskPulse 1.8s ease-in-out infinite;
}
@keyframes riskPulse {
    0%, 100% { box-shadow: 0 8px 24px rgba(255,77,94,.20); }
    50% { box-shadow: 0 10px 34px rgba(255,77,94,.42); }
}
.high-risk-card { border: 1px solid rgba(255,77,94,.65) !important; }
.high-risk-pulse { position: relative; }
.risk-low, .risk-medium { animation: badgeIn .45s ease-out; }
@keyframes badgeIn {
    from { opacity: 0; transform: translateY(8px) scale(.96); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

/* Responsive */
@media (max-width: 800px) {
    .block-container { padding: 1rem .8rem 2rem; }
    .title-glitch { font-size: 2.15rem; }
    .cyber-card { padding: 1rem; }
}

/* ===== CARDIORISK AI — DEEP PURPLE NEON THEME ===== */
:root {
  --bg:#09051b; --panel:#17102f; --panel2:#211342;
  --line:rgba(190,130,255,.22); --text:#f7f0ff; --muted:#c0afd9;
  --purple:#a855f7; --pink:#ec4899; --violet:#7c3aed;
  --green:#34d399; --amber:#fbbf24; --red:#fb5577;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif !important; }
.stApp {
 color:var(--text) !important;
 background:
   radial-gradient(ellipse at 78% 8%,rgba(126,34,206,.25),transparent 32%),
   radial-gradient(ellipse at 8% 35%,rgba(88,28,135,.24),transparent 34%),
   radial-gradient(ellipse at 50% 100%,rgba(76,29,149,.18),transparent 42%),
   linear-gradient(135deg,#080416 0%,#100824 48%,#09051b 100%) !important;
}
[data-testid="stHeader"] { background:rgba(9,5,27,.78) !important; }
[data-testid="stSidebar"] {
 background:linear-gradient(180deg,rgba(28,13,54,.98),rgba(12,7,34,.99)) !important;
 border:1px solid rgba(192,132,252,.22) !important;
 border-radius:24px !important;
 box-shadow:8px 0 45px rgba(0,0,0,.24),inset 0 1px 0 rgba(255,255,255,.05);
}
[data-testid="stSidebar"] * { color:#f2eaff !important; }
.block-container { max-width:1500px; padding-top:1.5rem; padding-bottom:3rem; }
h1,h2,h3,h4 { color:#fbf7ff !important; font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
p,label,.stMarkdown { color:#e9def8 !important; }
small,[data-testid="stCaptionContainer"] { color:var(--muted) !important; }
.title-glitch {
 font-family:'Space Grotesk',sans-serif !important; font-size:clamp(2.4rem,5vw,4.4rem);
 line-height:1.05; font-weight:800; text-align:center; letter-spacing:.025em;
 background:linear-gradient(90deg,#fff 5%,#f1e8ff 48%,#c084fc 75%,#e879f9 100%) !important;
 -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important;
 filter:drop-shadow(0 0 18px rgba(168,85,247,.28)); margin-bottom:.6rem;
}
.hero-subtitle { text-align:center; color:#d5c4ec !important; font-size:1.08rem; }
.author-credit {
 display:inline-block; margin:.9rem auto 0; padding:.72rem 1.45rem;
 border:1px solid rgba(216,180,254,.52); border-radius:999px;
 color:#f4eaff !important; font-weight:700; letter-spacing:.02em;
 background:linear-gradient(110deg,rgba(88,28,135,.38),rgba(30,16,58,.82));
 box-shadow:0 0 12px rgba(168,85,247,.35),0 0 30px rgba(217,70,239,.18),inset 0 1px 0 rgba(255,255,255,.12);
}
.author-credit strong {
 color:#fff !important; text-shadow:0 0 6px #d946ef,0 0 14px #a855f7,0 0 26px rgba(192,132,252,.85);
}
.heart-beat-png { filter:drop-shadow(0 0 15px rgba(236,72,153,.62)) drop-shadow(0 0 36px rgba(168,85,247,.38)) !important; }
.cyber-card,div[data-testid="stVerticalBlockBorderWrapper"] {
 background:linear-gradient(145deg,rgba(34,19,63,.88),rgba(19,12,43,.94)) !important;
 color:var(--text) !important; border:1px solid rgba(192,132,252,.22) !important;
 border-radius:24px !important; box-shadow:0 18px 55px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.06) !important;
}
[data-testid="stMetric"] {
 background:linear-gradient(145deg,rgba(48,27,83,.95),rgba(27,17,55,.95)) !important;
 border:1px solid var(--line) !important; border-radius:18px !important; padding:16px 18px;
 box-shadow:0 10px 28px rgba(0,0,0,.18);
}
[data-testid="stMetricLabel"] { color:#cbb8e5 !important; }
[data-testid="stMetricValue"] { color:#fff !important; font-family:'Space Grotesk',sans-serif; }
.stButton>button,.stDownloadButton>button {
 color:#fff !important; border:1px solid rgba(216,180,254,.42) !important;
 border-radius:14px !important; background:linear-gradient(100deg,#7c3aed,#a855f7 58%,#d946ef) !important;
 font-weight:800 !important; padding:.72rem 1.1rem;
 box-shadow:0 8px 24px rgba(168,85,247,.24) !important; transition:all .2s ease;
}
.stButton>button:hover,.stDownloadButton>button:hover { transform:translateY(-2px); box-shadow:0 12px 32px rgba(217,70,239,.36) !important; border-color:#e9d5ff !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
 background:#e879f9 !important; border:2px solid #fff !important;
 box-shadow:0 0 0 4px rgba(192,132,252,.18),0 0 16px rgba(217,70,239,.75) !important;
}
.stSlider [data-testid="stTickBar"] { background:rgba(192,132,252,.42) !important; }
.stTextInput input,.stNumberInput input,.stTextArea textarea,
.stSelectbox [data-baseweb="select"]>div,.stDateInput input {
 background:rgba(27,17,55,.96) !important; color:#fff !important;
 border:1px solid rgba(192,132,252,.32) !important; border-radius:12px !important;
}
[data-baseweb="tab-list"] { gap:.45rem; background:rgba(13,7,32,.58) !important; border:1px solid var(--line) !important; padding:.4rem; border-radius:16px; }
[data-baseweb="tab"] { color:#cbb8e5 !important; background:transparent !important; border-radius:12px !important; }
[data-baseweb="tab"][aria-selected="true"] { color:#fff !important; background:linear-gradient(100deg,#7c3aed,#a855f7) !important; box-shadow:0 0 18px rgba(168,85,247,.32); }
.stProgress>div>div>div>div { background:linear-gradient(90deg,#9333ea,#e879f9) !important; }
[data-testid="stAlert"] { background:rgba(35,20,64,.9) !important; color:#f7f0ff !important; border:1px solid var(--line) !important; border-radius:15px !important; }
[data-testid="stExpander"] { background:rgba(29,17,54,.72) !important; border:1px solid var(--line) !important; border-radius:14px !important; }
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label { color:#eadcf8 !important; }
hr { border-color:rgba(192,132,252,.20) !important; }
@keyframes purpleGlow { 0%,100%{filter:drop-shadow(0 0 10px rgba(168,85,247,.22));} 50%{filter:drop-shadow(0 0 22px rgba(217,70,239,.42));} }
.author-credit strong { animation:purpleGlow 2.8s ease-in-out infinite; }
@media(max-width:800px){ .block-container{padding:1rem .8rem 2rem;} .title-glitch{font-size:2.2rem;} .cyber-card{padding:1rem;} }
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms !important;animation-iteration-count:1 !important;}}
</style>
""", unsafe_allow_html=True)

# ---------------- FONCTIONS UTILITAIRES ----------------
@st.cache_resource
def load_heart_model():
    """Charge le modèle de prédiction"""
    try:
        # Essayer plusieurs chemins possibles pour le modèle
        app_dir = os.path.dirname(os.path.abspath(__file__))
        model_paths = [
            os.path.join(app_dir, "Model.pkl"),
            os.path.join(app_dir, "Model.pkl-finale"),
            os.path.join(app_dir, "model.pkl"),
            "Model.pkl",
            "Model.pkl-finale",
            "model.pkl",
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                return model
        
        return None
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {str(e)}")
        return None

def preprocess_data_for_model(patient_data, model):
    """Prétraite les données pour qu'elles correspondent au format attendu par le modèle"""
    # Créer une copie des données
    processed_data = patient_data.copy()
    
    # Vérifier et convertir la variable famhist selon le format attendu par le modèle
    # Le modèle peut s'attendre à des valeurs encodées (0/1) ou spécifiques
    if 'famhist' in processed_data:
        # Essayer différents formats
        if processed_data['famhist'] == "Absent":
            # Essayer 0
            processed_data['famhist'] = 0
        elif processed_data['famhist'] == "Present":
            # Essayer 1
            processed_data['famhist'] = 1
    
    # Vérifier les autres variables pour s'assurer qu'elles sont numériques
    numeric_vars = ['age', 'sbp', 'tobacco', 'ldl', 'adiposity', 'obesity', 'alcohol', 'typea']
    for var in numeric_vars:
        if var in processed_data:
            processed_data[var] = float(processed_data[var])
    
    return processed_data

def simulate_aging(base_values, target_age):
    """Simule le vieillissement avec mise à jour réaliste des paramètres"""
    current_age = base_values['age']
    age_diff = target_age - current_age
    
    simulated = base_values.copy()
    simulated['age'] = target_age
    # Simulation réaliste des changements avec l'âge
    simulated['sbp'] = min(200, simulated['sbp'] + age_diff * 0.6)
    simulated['ldl'] = min(10, simulated['ldl'] + age_diff * 0.04)
    simulated['adiposity'] = min(50, simulated['adiposity'] + age_diff * 0.2)
    simulated['obesity'] = min(45, simulated['obesity'] + age_diff * 0.15)
    simulated['typea'] = max(0, simulated['typea'] - age_diff * 0.25)
    
    # Tabac diminue avec l'âge (arrêt)
    if simulated['tobacco'] > 0:
        simulated['tobacco'] = max(0, simulated['tobacco'] - age_diff * 0.3)
    
    return simulated

def get_risk_factors(patient_data):
    """Analyse les facteurs de risque et retourne des recommandations"""
    factors = []
    
    if patient_data['age'] > 60:
        factors.append(("👴 Âge avancé", "danger", 
                       "Surveillance cardiaque régulière recommandée"))
    
    if patient_data['ldl'] > 4.0:
        level = "critique" if patient_data['ldl'] > 4.5 else "élevé"
        factors.append((f"🧬 LDL {level}", "danger" if patient_data['ldl'] > 4.5 else "warning",
                       f"Valeur LDL {patient_data['ldl']:.1f} - Objectif < 3.0"))
    
    if patient_data['tobacco'] > 10:
        factors.append(("🚬 Tabagisme important", "danger",
                       f"Consommation: {patient_data['tobacco']:.1f} kg/an"))
    
    if patient_data['sbp'] > 140:
        factors.append(("💓 Hypertension", "warning",
                       f"Pression systolique: {patient_data['sbp']} mmHg"))
    
    # Gérer différents formats de famhist
    famhist_value = patient_data.get('famhist', 'Absent')
    if isinstance(famhist_value, str):
        if famhist_value == "Present":
            factors.append(("👨‍👩‍👧 Antécédents familiaux", "warning",
                           "Facteur génétique identifié"))
    elif famhist_value == 1:
        factors.append(("👨‍👩‍👧 Antécédents familiaux", "warning",
                       "Facteur génétique identifié"))
    
    if patient_data['obesity'] > 30:
        factors.append(("⚖️ Obésité", "warning",
                       f"IMC estimé élevé: {patient_data['obesity']:.1f}"))
    
    if patient_data['alcohol'] > 30:
        factors.append(("🍷 Alcool élevé", "warning",
                       f"Consommation: {patient_data['alcohol']:.1f} g/semaine"))
    
    return factors

def calculate_health_score(patient_data):
    """Calcule un score de santé global"""
    score = 100
    
    # Pénalités selon les facteurs de risque
    if patient_data['age'] > 50:
        score -= (patient_data['age'] - 50) * 0.5
    
    if patient_data['ldl'] > 3.0:
        score -= (patient_data['ldl'] - 3.0) * 5
    
    if patient_data['tobacco'] > 0:
        score -= patient_data['tobacco'] * 1.5
    
    if patient_data['sbp'] > 120:
        score -= (patient_data['sbp'] - 120) * 0.3
    
    # Gérer différents formats de famhist
    famhist_value = patient_data.get('famhist', 'Absent')
    if (isinstance(famhist_value, str) and famhist_value == "Present") or famhist_value == 1:
        score -= 10
    
    if patient_data['obesity'] > 25:
        score -= (patient_data['obesity'] - 25) * 1
    
    if patient_data['alcohol'] > 20:
        score -= (patient_data['alcohol'] - 20) * 0.2
    
    return max(0, min(100, score))

def predict_with_model(model, patient_data):
    """Faire une prédiction avec le modèle en gérant les erreurs de format"""
    try:
        # Première tentative avec les données telles quelles
        X_new = pd.DataFrame([patient_data])
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_new)[0][1]
        else:
            pred = model.predict(X_new)[0]
            proba = pred
        
        return proba
    
    except ValueError as e:
        # Si erreur due au format des données, essayer avec prétraitement
        if "unknown categories" in str(e) or "could not convert" in str(e):
            st.warning("⚠️ Ajustement du format des données pour le modèle...")
            
            # Essayer différents formats pour famhist
            alternative_formats = [
                {'Absent': 0, 'Present': 1},
                {'Absent': '0', 'Present': '1'},
                {'Absent': 'No', 'Present': 'Yes'},
                {'Absent': 'absent', 'Present': 'present'},
            ]
            
            for fmt in alternative_formats:
                try:
                    test_data = patient_data.copy()
                    if 'famhist' in test_data:
                        test_data['famhist'] = fmt[test_data['famhist']]
                    
                    X_new = pd.DataFrame([test_data])
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_new)[0][1]
                    else:
                        pred = model.predict(X_new)[0]
                        proba = pred
                    
                    return proba
                    
                except:
                    continue
            
            # Si toutes les tentatives échouent, utiliser la simulation
            st.info("ℹ️ Utilisation de la simulation basée sur les facteurs de risque")
            return simulate_risk_score(patient_data)
        
        else:
            # Autre type d'erreur
            raise e

def simulate_risk_score(patient_data):
    """Simule un score de risque basé sur les facteurs de risque"""
    base_risk = 30
    
    # Ajustements selon les facteurs
    if patient_data['age'] > 60:
        base_risk += 15
    if patient_data['ldl'] > 4.0:
        base_risk += 20
    if patient_data['tobacco'] > 10:
        base_risk += 25
    if patient_data['sbp'] > 140:
        base_risk += 10
    
    # Gérer famhist
    famhist_value = patient_data.get('famhist', 'Absent')
    if (isinstance(famhist_value, str) and famhist_value == "Present") or famhist_value == 1:
        base_risk += 15
    
    if patient_data['obesity'] > 30:
        base_risk += 10
    if patient_data['alcohol'] > 30:
        base_risk += 5
    
    # Convertir en probabilité (0-1)
    risk_percent = min(100, base_risk)
    return risk_percent / 100

# ---------------- INITIALISATION SESSION ----------------
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
        'age': 45, 'sbp': 130, 'tobacco': 5.0,
        'ldl': 3.5, 'adiposity': 25.0, 'obesity': 27.0,
        'alcohol': 20.0, 'typea': 50, 'famhist': "Absent"
    }

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'show_high_risk_effects' not in st.session_state:
    st.session_state.show_high_risk_effects = False

if 'show_low_risk_effects' not in st.session_state:
    st.session_state.show_low_risk_effects = False

# ---------------- CHARGEMENT DU MODÈLE (SILENCIEUX) ----------------
model = load_heart_model()

# ---------------- SIDEBAR AVANCÉE ----------------
with st.sidebar:
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.markdown("## ⚙️ **Laboratoire d'Analyse**")
    
    analysis_mode = st.radio(
        "Mode d'analyse",
        ["🔍 Analyse Standard", "🎬 Simulation Temporelle", "📊 Analyse Comparative"],
        index=0
    )
    
    st.markdown("---")
    
    # Image de cœur à la place de Lottie
    heart_img = load_heart_image()
    if heart_img:
        heart_img_resized = heart_img.resize((150, 150))
        heart_base64 = get_base64_image(heart_img_resized)
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{heart_base64}" 
                 style="width: 150px; height: 150px; border-radius: 50%; 
                        animation: heartbeat 1.5s ease-in-out infinite;
                        filter: drop-shadow(0 0 15px rgba(255, 0, 0, 0.7));">
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Visualisations")
    show_pca = st.checkbox("Afficher l'analyse ACP", value=True)
    show_radar = st.checkbox("Afficher le radar santé", value=True)
    
    st.markdown("### ⚡ Effets Visuels")
    show_animations = st.checkbox("Activer les animations", value=True)
    
    st.markdown("---")
    st.markdown("### 💾 Historique")
    
    if st.button("🔄 Effacer l'historique", use_container_width=True):
        st.session_state.prediction_history = []
        st.success("Historique effacé !")
        st.rerun()
    
    if st.session_state.prediction_history:
        st.info(f"📊 {len(st.session_state.prediction_history)} analyses enregistrées")
    
    # Score de santé
    health_score = calculate_health_score(st.session_state.patient_data)
    st.metric("🎯 Score Santé", f"{health_score:.0f}/100")
    
    # État du modèle
    if model:
        st.success("✅ Modèle chargé et prêt")
        # Afficher des informations sur le modèle
        with st.expander("ℹ️ Info modèle"):
            try:
                if hasattr(model, 'steps'):
                    st.write(f"Type: Pipeline avec {len(model.steps)} étapes")
                elif hasattr(model, 'predict_proba'):
                    st.write("Type: Modèle avec probabilités")
                else:
                    st.write("Type: Modèle de classification")
            except:
                st.write("Type: Modèle chargé avec succès")
    else:
        st.warning("⚠️ Mode simulation - Modèle non chargé")
        with st.expander("ℹ️ Pour charger le modèle"):
            st.write("""
            1. Assurez-vous que le fichier `Model.pkl` est dans le dossier
            2. Vérifiez les permissions du fichier
            3. Redémarrez l'application
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- EN-TÊTE PRINCIPALE AVEC IMAGE CŒUR PNG QUI BAT ----------------
# Charger l'image du cœur
heart_img = load_heart_image()

# Créer un conteneur pour l'en-tête
st.markdown("""
<div class="header-container">
    <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
""", unsafe_allow_html=True)

# Colonnes pour l'image du cœur et le titre
col_heart, col_title = st.columns([1, 4])

with col_heart:
    if heart_img:
        # Redimensionner l'image
        heart_img_resized = heart_img.resize((180, 180))
        
        # Convertir en base64 pour l'affichage HTML
        heart_base64 = get_base64_image(heart_img_resized)
        
        # Afficher avec animation CSS de battement
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{heart_base64}" 
                 class="heart-beat-png"
                 style="width: 180px; height: 180px; border-radius: 50%;">
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback avec SVG animé
        heart_svg = """
        <svg width="180" height="180" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <style>
                .heart-path {
                    fill: #ef4444;
                    animation: heartbeat 1.5s ease-in-out infinite;
                    transform-origin: center;
                    filter: drop-shadow(0 0 15px rgba(239, 68, 68, 0.7));
                }
                @keyframes heartbeat {
                    0%, 100% { transform: scale(1); }
                    25% { transform: scale(1.1); }
                    50% { transform: scale(1.15); }
                    75% { transform: scale(1.1); }
                }
            </style>
            <path class="heart-path" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
        </svg>
        """
        st.markdown(f'<div style="text-align: center;">{heart_svg}</div>', unsafe_allow_html=True)

with col_title:
    st.markdown("<div class='title-glitch'>CARDIORISK AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle'>Intelligence artificielle & analyse des facteurs cardiovasculaires</div>",
                unsafe_allow_html=True)
    st.markdown("<div style='text-align:center'><div class='author-credit'>Réalisé par : <strong>MOHAMED AYOUB BOUBKRI</strong></div></div>",
                unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SIMULATION TEMPORELLE ----------------
if analysis_mode == "🎬 Simulation Temporelle":
    st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
    st.markdown("## ⏳ **Simulation de Vieillissement**")
    
    col_sim1, col_sim2 = st.columns([2, 1])
    
    with col_sim1:
        start_age = st.slider("Âge de départ", 20, 70, 30, key="start_age")
        end_age = st.slider("Âge d'arrivée", start_age + 10, 80, 70, key="end_age")
        steps = st.slider("Nombre d'étapes", 5, 20, 10)
    
    with col_sim2:
        st.markdown("### Paramètres initiaux")
        tobacco_base = st.slider("Tabac initial (kg/an)", 0.0, 30.0, 5.0, key="tobacco_base")
        ldl_base = st.slider("LDL initial (mmol/L)", 1.0, 8.0, 3.0, key="ldl_base")
    
    if st.button("🚀 Lancer la simulation temporelle", key="simulate_btn", use_container_width=True):
        # Initialisation
        st.session_state.patient_data['age'] = start_age
        st.session_state.patient_data['tobacco'] = tobacco_base
        st.session_state.patient_data['ldl'] = ldl_base
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        simulation_results = []
        
        # Graphique en temps réel
        chart_placeholder = st.empty()
        
        # Simulation
        ages = np.linspace(start_age, end_age, steps)
        
        for i, target_age in enumerate(ages):
            # Simulation du vieillissement
            simulated = simulate_aging(st.session_state.patient_data, int(target_age))
            st.session_state.patient_data.update(simulated)
            
            # Calcul du score de santé
            health_score = calculate_health_score(st.session_state.patient_data)
            simulation_results.append({
                'Âge': int(target_age),
                'Score Santé': health_score,
                'Pression': simulated['sbp'],
                'LDL': simulated['ldl']
            })
            
            # Mise à jour du graphique
            if simulation_results:
                df_sim = pd.DataFrame(simulation_results)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_sim['Âge'], y=df_sim['Score Santé'],
                    mode='lines+markers',
                    name='Score Santé',
                    line=dict(color='#00f3ff', width=4),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Évolution du Score de Santé avec l'Âge",
                    xaxis_title="Âge",
                    yaxis_title="Score Santé",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    hovermode='x unified'
                )
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            
            # Mise à jour de la progression
            progress = (i + 1) / len(ages)
            progress_bar.progress(progress)
            status_text.text(f"⏳ Simulation âge {int(target_age)} ans...")
            time.sleep(0.2)
        
        status_text.success("✅ Simulation terminée !")
        st.balloons()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INTERFACE PRINCIPALE DES DONNÉES ----------------
st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
st.markdown("## 🧬 **Profil Biomédical du Patient**")

tab1, tab2, tab3 = st.tabs(["📊 Données Vitales", "🎚️ Contrôle Précis", "📈 Visualisations"])

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.patient_data['age'] = st.slider(
            "**Âge (ans)**", 20, 80, st.session_state.patient_data['age'],
            help="Facteur de risque majeur - Augmente avec l'âge"
        )
        
        st.session_state.patient_data['sbp'] = st.slider(
            "**Pression Systolique (mmHg)**", 90, 200, st.session_state.patient_data['sbp'],
            help="Pression artérielle systolique - Objectif < 140 mmHg"
        )
        
        st.session_state.patient_data['tobacco'] = st.slider(
            "**Tabac (kg/an)**", 0.0, 50.0, st.session_state.patient_data['tobacco'],
            help="Consommation annuelle cumulée - Facteur de risque majeur"
        )
    
    with col2:
        st.session_state.patient_data['ldl'] = st.slider(
            "**LDL Cholestérol (mmol/L)**", 0.5, 10.0, st.session_state.patient_data['ldl'],
            help="Mauvais cholestérol - Objectif < 3.0 mmol/L"
        )
        
        st.session_state.patient_data['adiposity'] = st.slider(
            "**Adiposité (%)**", 5.0, 50.0, st.session_state.patient_data['adiposity'],
            help="Pourcentage de masse graisseuse corporelle"
        )
        
        st.session_state.patient_data['obesity'] = st.slider(
            "**Indice Obésité**", 15.0, 45.0, st.session_state.patient_data['obesity'],
            help="Indice de masse grasse - Objectif < 30"
        )
    
    with col3:
        st.session_state.patient_data['alcohol'] = st.slider(
            "**Alcool (g/semaine)**", 0.0, 150.0, st.session_state.patient_data['alcohol'],
            help="Consommation hebdomadaire - Recommandation < 140g"
        )
        
        st.session_state.patient_data['typea'] = st.slider(
            "**Stress (Type A)**", 0, 100, st.session_state.patient_data['typea'],
            help="Comportement type A - Stress et compétitivité"
        )
        
        st.session_state.patient_data['famhist'] = st.selectbox(
            "**Antécédents Familiaux**", 
            ["Absent", "Present"],
            index=0 if st.session_state.patient_data['famhist'] == "Absent" else 1,
            help="Présence d'antécédents cardiaques familiaux"
        )

with tab2:
    col_prec1, col_prec2 = st.columns(2)
    
    with col_prec1:
        st.markdown("### 🔬 Valeurs Exactes")
        st.session_state.patient_data['ldl'] = st.number_input(
            "LDL exact (mmol/L)", 0.5, 10.0, st.session_state.patient_data['ldl'], 0.1,
            format="%.1f"
        )
        st.session_state.patient_data['sbp'] = st.number_input(
            "Pression exacte (mmHg)", 90, 200, st.session_state.patient_data['sbp'], 1
        )
        st.session_state.patient_data['age'] = st.number_input(
            "Âge exact (ans)", 20, 80, st.session_state.patient_data['age'], 1
        )
    
    with col_prec2:
        st.markdown("### 🎯 Cibles de Santé")
        
        # Indicateurs de santé
        col_indic1, col_indic2, col_indic3 = st.columns(3)
        
        with col_indic1:
            ldl_status = "✅" if st.session_state.patient_data['ldl'] < 3.0 else "⚠️"
            st.metric("LDL", f"{st.session_state.patient_data['ldl']:.1f}", 
                     delta=ldl_status)
        
        with col_indic2:
            sbp_status = "✅" if st.session_state.patient_data['sbp'] < 140 else "⚠️"
            st.metric("Pression", f"{st.session_state.patient_data['sbp']}", 
                     delta=sbp_status)
        
        with col_indic3:
            tobacco_status = "✅" if st.session_state.patient_data['tobacco'] == 0 else "⚠️"
            st.metric("Tabac", f"{st.session_state.patient_data['tobacco']:.1f}", 
                     delta=tobacco_status)

with tab3:
    if show_radar:
        # Diagramme radar des facteurs de risque
        categories = ['Âge', 'Pression', 'Tabac', 'LDL', 'Adiposité', 'Alcool']
        
        # Normalisation des valeurs pour le radar
        values = [
            min(st.session_state.patient_data['age'] / 80 * 100, 100),
            min(st.session_state.patient_data['sbp'] / 200 * 100, 100),
            min(st.session_state.patient_data['tobacco'] / 50 * 100, 100),
            min(st.session_state.patient_data['ldl'] / 10 * 100, 100),
            min(st.session_state.patient_data['adiposity'] / 50 * 100, 100),
            min(st.session_state.patient_data['alcohol'] / 150 * 100, 100)
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Profil actuel',
            fillcolor='rgba(0, 243, 255, 0.3)',
            line=dict(color='#00f3ff', width=3)
        ))
        
        # Ajouter la cible idéale
        target_values = [30, 70, 0, 30, 50, 20]  # Valeurs cibles
        fig_radar.add_trace(go.Scatterpolar(
            r=target_values,
            theta=categories,
            fill='toself',
            name='Cible idéale',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color='white')
                ),
                angularaxis=dict(tickfont=dict(color='white')),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=True,
            legend=dict(
                font=dict(color='white'),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BOUTON DE PRÉDICTION PRINCIPAL ----------------
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_button = st.button(
        "🚀 **ANALYSER LE RISQUE CARDIAQUE** 🚀",
        key="main_predict",
        use_container_width=True,
        type="primary"
    )

# ---------------- PRÉDICTION ET VISUALISATION ----------------
if predict_button:
    # Appliquer les effets selon le niveau de risque
    if st.session_state.show_high_risk_effects:
        st.markdown("<div class='high-risk-card'>", unsafe_allow_html=True)
    elif st.session_state.show_low_risk_effects:
        st.markdown("<div class='confetti-effect'>", unsafe_allow_html=True)
    
    with st.spinner("🔬 **Analyse médicale avancée en cours...**"):
        # Animation de chargement
        progress_bar = st.progress(0)
        status_ph = st.empty()
        
        steps = [
            "🔍 Scanner des paramètres biomédicaux...",
            "🧬 Analyse des marqueurs génétiques...",
            "📊 Calcul des corrélations multi-factorielles...",
            "🤖 Application de l'algorithme d'IA...",
            "📈 Génération du rapport personnalisé..."
        ]
        
        for i, step in enumerate(steps):
            status_ph.text(f"{step}")
            progress_bar.progress((i + 1) * 20)
            time.sleep(0.4)
        
        try:
            # Vérifier si le modèle est chargé
            if model:
                # Essayer de faire la prédiction avec le modèle
                proba = predict_with_model(model, st.session_state.patient_data)
                risk_percent = proba * 100
                model_used = "Modèle d'IA"
            else:
                # Simulation si modèle non chargé
                st.warning("⚠️ Modèle non chargé - Utilisation de la simulation avancée")
                risk_percent = simulate_risk_score(st.session_state.patient_data) * 100
                model_used = "Simulation avancée"
            
            # Sauvegarde dans l'historique
            history_entry = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'risk_percent': risk_percent,
                'age': st.session_state.patient_data['age'],
                'ldl': st.session_state.patient_data['ldl'],
                'sbp': st.session_state.patient_data['sbp'],
                'model': model_used
            }
            st.session_state.prediction_history.append(history_entry)
            
            # ---------------- AFFICHAGE DES RÉSULTATS ----------------
            st.markdown("---")
            
            # Détermination du niveau de risque
            if risk_percent < 25:
                risk_level = "low"
                color = "#10b981"
                emoji = "🟢"
                risk_text = "FAIBLE RISQUE"
                badge_class = "risk-low"
                message = "Excellent profil cardiovasculaire !"
                st.session_state.show_high_risk_effects = False
                st.session_state.show_low_risk_effects = True
            elif risk_percent < 60:
                risk_level = "medium"
                color = "#f59e0b"
                emoji = "🟡"
                risk_text = "RISQUE MODÉRÉ"
                badge_class = "risk-medium"
                message = "Surveillance recommandée avec amélioration du style de vie."
                st.session_state.show_high_risk_effects = False
                st.session_state.show_low_risk_effects = False
            else:
                risk_level = "high"
                color = "#ef4444"
                emoji = "🔴"
                risk_text = "RISQUE ÉLEVÉ"
                badge_class = "risk-high high-risk-pulse"
                message = "Consulter un cardiologue rapidement !"
                st.session_state.show_high_risk_effects = True
                st.session_state.show_low_risk_effects = False
            
            # Affichage du résultat principal
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            with col_res2:
                st.markdown(f"""
                <div style='text-align: center;'>
                    <div style='font-size: 5rem; margin-bottom: 1rem;'>{emoji}</div>
                    <div class='risk-badge {badge_class}'>
                        {risk_text}
                    </div>
                    <h2 style='color: {color}; margin-top: 1rem; font-size: 2.5rem;'>
                        {risk_percent:.1f}% de probabilité
                    </h2>
                    <p style='color: #94a3b8; font-size: 1.1rem;'>
                        {message}
                    </p>
                    <p style='color: #64748b; font-size: 0.9rem; margin-top: 10px;'>
                        <i>Calculé avec: {model_used}</i>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # EFFETS VISUELS SPÉCIAUX
            if show_animations:
                if risk_level == "high":
                    st.markdown("""
                    <div style='text-align: center; margin: 20px 0;'>
                        <div style='color: #ef4444; font-size: 1.2rem; font-weight: bold; animation: highRiskAlert 1.5s ease-in-out infinite;'>
                            ⚠️ ALERTE MÉDICALE - CONSULTATION URGENTE RECOMMANDÉE ⚠️
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                elif risk_level == "low":
                    st.markdown("""
                    <div style='text-align: center; margin: 20px 0;'>
                        <div style='color: #10b981; font-size: 1.2rem; font-weight: bold;'>
                            🎉 FÉLICITATIONS ! PROFIL CARDIAQUE OPTIMAL 🎉
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # ---------------- JAUGE DE RISQUE ----------------
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("## 📊 **Analyse Quantitative du Risque**")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_percent,
                number={'suffix': "%", 'font': {'size': 60, 'color': color}},
                delta={'reference': 50, 'increasing': {'color': color}},
                gauge={
                    'axis': {
                        'range': [0, 100], 
                        'tickwidth': 2,
                        'tickcolor': "white",
                        'tickfont': {'color': 'white'}
                    },
                    'bar': {'color': color, 'thickness': 0.4},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.3)'},
                        {'range': [25, 60], 'color': 'rgba(245, 158, 11, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 5},
                        'thickness': 0.9,
                        'value': risk_percent
                    }
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            
            fig_gauge.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Exo 2"},
                margin=dict(t=50, b=50, l=50, r=50)
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- FACTEURS DE RISQUE DÉTAILLÉS ----------------
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("## ⚠️ **Facteurs de Risque Identifiés**")
            
            risk_factors = get_risk_factors(st.session_state.patient_data)
            
            if risk_factors:
                cols = st.columns(2)
                for idx, (factor, level, description) in enumerate(risk_factors):
                    with cols[idx % 2]:
                        if level == "danger":
                            st.error(f"**{factor}**\n\n{description}")
                        else:
                            st.warning(f"**{factor}**\n\n{description}")
            else:
                st.success("✅ Aucun facteur de risque majeur identifié !")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- RECOMMANDATIONS PERSONNALISÉES ----------------
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("## 🎯 **Plan d'Action Personnalisé**")
            
            # Timeline d'action
            timeline_col1, timeline_col2, timeline_col3 = st.columns(3)
            
            with timeline_col1:
                st.markdown("### ⏱️ **Immédiat (1-7 jours)**")
                st.write("✅ Consultation médicale")
                st.write("✅ Bilan sanguin complet")
                st.write("✅ Mesure tensionnelle quotidienne")
                
                if risk_level == "high":
                    st.error("**URGENT:** Consultation cardiologique")
            
            with timeline_col2:
                st.markdown("### 🗓️ **Court terme (1-3 mois)**")
                st.write("✅ Programme activité physique")
                st.write("✅ Adaptation nutritionnelle")
                st.write("✅ Suivi paramétrique")
                st.write("✅ Gestion du stress")
            
            with timeline_col3:
                st.markdown("### 🗓️ **Moyen terme (3-12 mois)**")
                st.write("✅ Atteinte objectifs biologiques")
                st.write("✅ Maintien des habitudes saines")
                st.write("✅ Contrôle régulier")
                st.write("✅ Éducation thérapeutique")
            
            st.markdown("---")
            
            # Objectifs chiffrés
            st.markdown("### 🎯 **Objectifs Chiffrés**")
            
            obj_col1, obj_col2, obj_col3, obj_col4 = st.columns(4)
            
            with obj_col1:
                current_ldl = st.session_state.patient_data['ldl']
                target_ldl = min(3.0, current_ldl * 0.8)
                st.metric("LDL Cible", f"{target_ldl:.1f}", 
                         delta=f"-{current_ldl-target_ldl:.1f}")
            
            with obj_col2:
                current_sbp = st.session_state.patient_data['sbp']
                target_sbp = min(140, current_sbp * 0.9)
                st.metric("Pression Cible", f"{target_sbp:.0f}", 
                         delta=f"-{current_sbp-target_sbp:.0f}")
            
            with obj_col3:
                current_tobacco = st.session_state.patient_data['tobacco']
                target_tobacco = 0
                st.metric("Tabac Cible", f"{target_tobacco:.0f}", 
                         delta=f"-{current_tobacco:.1f}")
            
            with obj_col4:
                current_score = calculate_health_score(st.session_state.patient_data)
                target_score = min(100, current_score + 20)
                st.metric("Score Santé Cible", f"{target_score:.0f}", 
                         delta=f"+{target_score-current_score:.0f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- HISTORIQUE DES ANALYSES ----------------
            if len(st.session_state.prediction_history) > 1:
                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                st.markdown("## 📈 **Évolution de Votre Risque**")
                
                history_df = pd.DataFrame(st.session_state.prediction_history)
                
                fig_history = px.line(
                    history_df, 
                    x='timestamp', 
                    y='risk_percent',
                    title="Historique des analyses de risque",
                    markers=True,
                    line_shape='spline'
                )
                
                # Ajouter des zones de couleur selon le niveau de risque
                fig_history.add_hrect(
                    y0=60, y1=100,
                    fillcolor="rgba(239, 68, 68, 0.2)",
                    layer="below",
                    line_width=0
                )
                
                fig_history.add_hrect(
                    y0=25, y1=60,
                    fillcolor="rgba(245, 158, 11, 0.2)",
                    layer="below",
                    line_width=0
                )
                
                fig_history.add_hrect(
                    y0=0, y1=25,
                    fillcolor="rgba(16, 185, 129, 0.2)",
                    layer="below",
                    line_width=0
                )
                
                fig_history.update_traces(
                    line=dict(color='#00f3ff', width=4),
                    marker=dict(size=8, color='white')
                )
                
                fig_history.update_layout(
                    xaxis_title="Date d'analyse",
                    yaxis_title="Risque (%)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_history, use_container_width=True)
                
                # Statistiques d'évolution
                if len(history_df) > 1:
                    risk_change = history_df['risk_percent'].iloc[-1] - history_df['risk_percent'].iloc[0]
                    st.metric(
                        "Évolution du risque",
                        f"{history_df['risk_percent'].iloc[-1]:.1f}%",
                        delta=f"{risk_change:+.1f}%"
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- RAPPORT MÉDICAL ----------------
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("## 📄 **Rapport Médical Complet**")
            
            # Préparer les données pour le rapport
            famhist_display = st.session_state.patient_data['famhist']
            if famhist_display == "Absent":
                famhist_text = "Absent"
            elif famhist_display == "Present":
                famhist_text = "Présent"
            elif famhist_display == 0:
                famhist_text = "Absent"
            elif famhist_display == 1:
                famhist_text = "Présent"
            else:
                famhist_text = str(famhist_display)
            
            # Génération du rapport
            report_content = f"""
            RAPPORT CARDIORISK AI - ANALYSE CARDIOLOGIQUE
            {'='*60}
            
            Date d'analyse: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            Niveau de risque: {risk_text} ({risk_percent:.1f}%)
            Méthode de calcul: {model_used}
            
            PROFIL PATIENT:
            {'-'*40}
            • Âge: {st.session_state.patient_data['age']} ans
            • Pression systolique: {st.session_state.patient_data['sbp']} mmHg
            • LDL cholestérol: {st.session_state.patient_data['ldl']:.1f} mmol/L
            • Consommation tabac: {st.session_state.patient_data['tobacco']:.1f} kg/an
            • Adiposité: {st.session_state.patient_data['adiposity']:.1f}%
            • Obésité: {st.session_state.patient_data['obesity']:.1f}
            • Consommation alcool: {st.session_state.patient_data['alcohol']:.1f} g/semaine
            • Stress (Type A): {st.session_state.patient_data['typea']}/100
            • Antécédents familiaux: {famhist_text}
            
            FACTEURS DE RISQUE IDENTIFIÉS:
            {'-'*40}
            """
            
            for factor, level, description in risk_factors:
                report_content += f"• {factor}: {description}\n"
            
            report_content += f"""
            RECOMMANDATIONS MÉDICALES:
            {'-'*40}
            1. Consultation {('urgente' if risk_level == 'high' else 'programmée')}
            2. Bilan cardiologique complet
            3. Suivi des paramètres biologiques
            4. Adaptation du style de vie
            5. Éducation thérapeutique
            
            OBJECTIFS À 3 MOIS:
            {'-'*40}
            • LDL cholestérol: < 3.0 mmol/L
            • Pression artérielle: < 140/90 mmHg
            • Arrêt total du tabac
            • Activité physique: 150 min/semaine
            • Perte de poids si nécessaire
            
            NOTE IMPORTANTE:
            {'-'*40}
            Ce rapport est généré par l'algorithme CardioRisk AI.
            Il ne remplace pas une consultation médicale professionnelle.
            Consultez toujours un médecin pour interprétation et suivi.
            
            Score de santé global: {calculate_health_score(st.session_state.patient_data):.0f}/100
            """
            
            # Téléchargement du rapport
            st.download_button(
                label="💾 Télécharger le rapport complet (.TXT)",
                data=report_content,
                file_name=f"rapport_cardio_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            # Affichage du rapport dans un expander
            with st.expander("📋 Aperçu du rapport"):
                st.text(report_content)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # ---------------- CÉLÉBRATION / ALERTE FINALE ----------------
            if risk_level == "low":
                # Effet confetti et ballons
                st.balloons()
                st.snow()
                st.success("🎉 **Excellent ! Votre profil cardiovasculaire est optimal.**")
            
            elif risk_level == "medium":
                st.info("📋 **Surveillance recommandée.** Améliorez votre style de vie pour réduire les risques.")
            
            else:  # high risk
                st.error("🚨 **Risque élevé détecté !** Consultation médicale urgente requise.")
            
            status_ph.success(f"✅ **Analyse terminée** - {risk_text}")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
            st.info("""
            **Dépannage :**
            1. Vérifiez que votre modèle est compatible avec les données d'entrée
            2. Assurez-vous que le fichier modèle n'est pas corrompu
            3. Essayez de recréer le modèle avec le même format de données
            4. Utilisez le mode simulation si le problème persiste
            """)
    
    # Fermer les divs d'effets si ouverts
    if st.session_state.show_high_risk_effects:
        st.markdown("</div>", unsafe_allow_html=True)
    elif st.session_state.show_low_risk_effects:
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**🔬 CardioRisk AI v2.1**")
    st.caption("Prototype académique — non certifié médicalement")
    st.caption("Performance à valider sur un jeu de test indépendant")

with footer_col2:
    st.markdown("**📚 Projet académique**")
    st.caption("HESTIM 2025-2026")
    st.caption("Machine Learning Avancé")

with footer_col3:
    st.markdown("**⚖️ Avertissement**")
    st.caption("À des fins éducatives")
    st.caption("© 2026 — Projet éducatif")