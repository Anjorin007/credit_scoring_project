import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import cohere
import io
from fpdf import FPDF
import unicodedata
import os

# -------------------------------
# Fonction pour retirer les accents (pour PDF)
# -------------------------------
def remove_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

# -------------------------------
# Dictionnaire de traduction
# -------------------------------
T = {
    "fr": {
        "page_title": "RiskScore Pro | Analyse de Crédit",
        "about": "Application professionnelle d’analyse du risque crédit basée sur l’IA.",
        "header_title": "RiskScore Pro - Évaluation du Risque Crédit",
        "header_subtitle": "Plateforme d’analyse prédictive fondée sur XGBoost, SHAP & IA générative",
        "profile_client": "Profil Client",
        "age": "Âge",
        "income": "Revenu mensuel (€)",
        "dependents": "Personnes à charge",
        "open_credit": "Crédits actifs",
        "real_estate": "Prêts immobiliers",
        "debt_ratio": "Ratio d'endettement",
        "revolving": "Utilisation crédit renouvelable (%)",
        "late_30": "Retards 30-59 jours",
        "late_60": "Retards 60-89 jours",
        "late_90": "Retards ≥90 jours",
        "analysis_result": "Résultat de l'analyse",
        "default_prob": "Probabilité de défaut",
        "recommendation": "Recommandation",
        "low_risk": "Faible risque",
        "high_risk": "Risque élevé",
        "acceptance": "Acceptation",
        "rejection": "Rejet",
        "low_risk_badge": "✅ CLIENT FAIBLE RISQUE",
        "high_risk_badge": "⚠ CLIENT À RISQUE ÉLEVÉ",
        "shap_section": "Analyse d'impact des variables (SHAP)",
        "interpretation": "🗣️ Interprétation en langage naturel",
        "explanation_high": "Sur la base des données fournies, le modèle considère que ce client présente un **risque élevé de défaut**. Cela est principalement dû à :",
        "explanation_low": "Sur la base des données fournies, le modèle considère que ce client est **fiable et présente un faible risque**. Cela est principalement dû à :",
        "conclusion_high": "🔍 **Conclusion** : Le client montre des signes clairs de difficulté financière passée.\nUne surveillance renforcée, une demande de justificatifs ou un encadrement du crédit sont recommandés.",
        "conclusion_low": "✅ **Conclusion** : Le profil du client est globalement sain.\nIl peut être éligible à une offre de crédit standard sous réserve de validation manuelle finale.",
        "ai_report_section": "Rapport d'analyse complet par IA",
        "generate_ai_report": "Générer le rapport IA",
        "missing_api_key": "⚠ Clé API Cohere manquante dans st.secrets.",
        "ai_in_progress": "Analyse IA en cours...",
        "pdf_button": "📄 Télécharger le rapport PDF",
        "excel_button": "📊 Télécharger les données Excel",
        "pdf_title": "RiskScore Pro - Rapport IA",
        "page": "Page",
        "prompt_template": """
Tu es un expert en analyse de crédit bancaire. Fournis un rapport synthétique, factuel et professionnel sur ce client :

Âge : {age} ans
Revenu mensuel : {income} €
Ratio d'endettement : {debt_ratio}
Utilisation crédit renouvelable : {revolving}%
Crédits actifs : {open_credit}
Prêts immobiliers : {real_estate}
Personnes à charge : {dependents}
Retards 30-59j : {late_30}
Retards 60-89j : {late_60}
Retards ≥90j : {late_90}
Probabilité de défaut : {proba:.2%}

Ne pose pas de questions, ne donne pas de conseils, limite-toi à l'analyse et la conclusion. Tout ça en 250 caractères maximum.
"""
    },
    "en": {
        "page_title": "RiskScore Pro | Credit Analysis",
        "about": "Professional application for credit risk analysis based on AI.",
        "header_title": "RiskScore Pro - Credit Risk Assessment",
        "header_subtitle": "Predictive analytics platform powered by XGBoost, SHAP & Generative AI",
        "profile_client": "Client Profile",
        "age": "Age",
        "income": "Monthly income (€)",
        "dependents": "Number of dependents",
        "open_credit": "Open credit lines",
        "real_estate": "Real estate loans",
        "debt_ratio": "Debt ratio",
        "revolving": "Revolving credit utilization (%)",
        "late_30": "30-59 days late payments",
        "late_60": "60-89 days late payments",
        "late_90": "≥90 days late payments",
        "analysis_result": "Analysis Result",
        "default_prob": "Probability of Default",
        "recommendation": "Recommendation",
        "low_risk": "Low risk",
        "high_risk": "High risk",
        "acceptance": "Approval",
        "rejection": "Rejection",
        "low_risk_badge": "✅ LOW-RISK CLIENT",
        "high_risk_badge": "⚠ HIGH-RISK CLIENT",
        "shap_section": "Variable Impact Analysis (SHAP)",
        "interpretation": "🗣️ Natural Language Interpretation",
        "explanation_high": "Based on the provided data, the model considers this client to have a **high risk of default**. This is mainly due to:",
        "explanation_low": "Based on the provided data, the model considers this client to be **reliable with low risk**. This is mainly due to:",
        "conclusion_high": "🔍 **Conclusion**: The client shows clear signs of past financial distress.\nEnhanced monitoring, documentation, or credit restrictions are recommended.",
        "conclusion_low": "✅ **Conclusion**: The client's profile is overall healthy.\nThey may be eligible for a standard credit offer pending final manual review.",
        "ai_report_section": "Complete Analysis Report by AI",
        "generate_ai_report": "Generate AI Report",
        "missing_api_key": "⚠ Missing Cohere API key in st.secrets.",
        "ai_in_progress": "AI analysis in progress...",
        "pdf_button": "📄 Download PDF report",
        "excel_button": "📊 Download Excel data",
        "pdf_title": "RiskScore Pro - AI Report",
        "page": "Page",
        "prompt_template": """
You are a banking credit analysis expert. Provide a concise, factual, and professional report about this client:

Age: {age} years
Monthly income: {income} €
Debt ratio: {debt_ratio}
Revolving credit utilization: {revolving}%
Open credit lines: {open_credit}
Real estate loans: {real_estate}
Dependents: {dependents}
30-59 days late payments: {late_30}
60-89 days late payments: {late_60}
≥90 days late payments: {late_90}
Probability of default: {proba:.2%}

Do not ask questions, do not provide advice, limit yourself to the analysis and conclusion. All this in a maximum of 250 characters.
"""
    }
}

# -------------------------------
# Chargement modèle
# -------------------------------
model = joblib.load("../models/xgb_model.pkl")

# -------------------------------
# Détection langue via URL ?lang=fr ou ?lang=en
# -------------------------------
params = st.query_params
lang_candidate = params.get("lang", ["fr"])
if isinstance(lang_candidate, list):
    lang_candidate = lang_candidate[0].lower()
else:
    lang_candidate = lang_candidate.lower()
lang = lang_candidate if lang_candidate in ["fr", "en"] else "fr"
tr = T[lang]

# -------------------------------
# Configuration page
# -------------------------------
st.set_page_config(
    page_title=tr["page_title"],
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": tr["about"]}
)

# -------------------------------
# CSS thème sombre simple
# -------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');
    * {font-family: 'Montserrat', sans-serif !important; color: #e0e6f2 !important;}
    body, .stApp, .main, .block-container {background-color: #121820 !important;}
    .header {background-color: #1f2937; padding: 2rem; border-radius: 0 0 30px 30px; text-align: center;}
    .header h1 {color: #38bdf8; font-size: 2.6rem; font-weight: 700; margin-bottom: 0.5rem;}
    .header p {color: #94a3b8; font-size: 1.1rem; margin-top: 0;}
    .metric-container {background: #1e293b; border-radius: 15px; padding: 1rem; margin-bottom: 1rem; text-align:center;}
    .metric-label {font-weight: 600; color: #94a3b8; font-size: 1rem;}
    .metric-value {font-weight: 700; font-size: 2rem; color: #38bdf8;}
    .risk-badge {display:inline-block; padding:0.7rem 2rem; border-radius:50px; font-weight:700; font-size:1.3rem; margin:2rem auto; user-select:none; box-shadow: 0 0 20px rgba(0,0,0,0.5);}
    .risk-low {background:#0f766e; color:#a7f3d0; text-shadow:0 0 8px #0f766e;}
    .risk-high {background:#991b1b; color:#fecaca; text-shadow:0 0 10px #991b1b;}
    .section-title {font-weight:700; font-size:1.8rem; margin:2rem 0 1rem 0; border-bottom:2px solid #2563eb; padding-bottom:0.5rem;}
    .shap-plot {background:#1e293b; border-radius:15px; padding:1rem; box-shadow:0 8px 24px rgba(0,0,0,0.8); margin-bottom:2rem;}
    .card {background:#1e293b; padding:1rem; border-radius:15px; color:#d1d5db; white-space: pre-wrap;}
    button[kind="primary"] {background-color:#2563eb !important; color:white !important; font-weight:600 !important; border-radius:12px !important; padding:0.7rem 1.5rem !important;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header
# -------------------------------
st.markdown(f"""
<div class="header">
    <h1>{tr['header_title']}</h1>
    <p>{tr['header_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Formulaire
# -------------------------------
with st.container():
    st.markdown(f'<div class="section-title">{tr["profile_client"]}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        age = st.slider(tr["age"], 18, 100, 35)
        income = st.number_input(tr["income"], min_value=0, value=3000, step=100)
        dependents = st.slider(tr["dependents"], 0, 10, 2)
        open_credit = st.slider(tr["open_credit"], 0, 30, 5)
        real_estate = st.slider(tr["real_estate"], 0, 10, 1)

    with col2:
        debt_ratio = st.slider(tr["debt_ratio"], 0.0, 10.0, 0.5, step=0.1)
        revolving = st.slider(tr["revolving"], 0.0, 100.0, 30.0, step=1.0)
        late_30 = st.slider(tr["late_30"], 0, 10, 0)
        late_60 = st.slider(tr["late_60"], 0, 10, 0)
        late_90 = st.slider(tr["late_90"], 0, 10, 0)

# -------------------------------
# Préparation données et prédiction
# -------------------------------
client_df = pd.DataFrame([{
    'RevolvingUtilizationOfUnsecuredLines': revolving / 100,
    'age': age,
    'NumberOfTime30-59DaysPastDueNotWorse': late_30,
    'DebtRatio': debt_ratio,
    'MonthlyIncome': income,
    'NumberOfOpenCreditLinesAndLoans': open_credit,
    'NumberOfTimes90DaysLate': late_90,
    'NumberRealEstateLoansOrLines': real_estate,
    'NumberOfTime60-89DaysPastDueNotWorse': late_60,
    'NumberOfDependents': dependents
}])

proba = model.predict_proba(client_df)[0][1]
classe = model.predict(client_df)[0]

# -------------------------------
# Affichage résultats
# -------------------------------
st.markdown(f'<div class="section-title">{tr["analysis_result"]}</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown(f'<div class="metric-container"><div class="metric-label">{tr["default_prob"]}</div><div class="metric-value">{proba:.2%}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-container"><div class="metric-label">{tr["income"]}</div><div class="metric-value">{income:,} €</div></div>', unsafe_allow_html=True)
with col3:
    status_text = tr["acceptance"] if classe == 0 else tr["rejection"]
    delta_text = tr["low_risk"] if classe == 0 else tr["high_risk"]
    delta_color = "#22c55e" if classe == 0 else "#ef4444"
    st.markdown(f'''
        <div class="metric-container" style="color: {delta_color};">
            <div class="metric-label">{tr["recommendation"]}</div>
            <div class="metric-value">{status_text}</div>
            <div class="metric-delta">{delta_text}</div>
        </div>
    ''', unsafe_allow_html=True)

if classe == 0:
    st.markdown(f'<div class="risk-badge risk-low">{tr["low_risk_badge"]}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="risk-badge risk-high">{tr["high_risk_badge"]}</div>', unsafe_allow_html=True)

# -------------------------------
# Analyse SHAP
# -------------------------------
st.markdown(f'<div class="section-title">{tr["shap_section"]}</div>', unsafe_allow_html=True)
explainer = shap.Explainer(model)
shap_values = explainer(client_df)

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.waterfall(shap_values[0], max_display=10)
plt.tight_layout()
st.pyplot(fig)

shap_df = pd.DataFrame({
    'feature': client_df.columns,
    'shap_value': shap_values[0].values,
    'value': client_df.iloc[0].values
}).sort_values(by='shap_value', key=abs, ascending=False)

top_features = shap_df.head(3)

explications = []
for _, row in top_features.iterrows():
    sens = (tr["high_risk"] if row['shap_value'] > 0 else tr["low_risk"]).lower()
    explications.append(f"- **{row['feature']}** ({row['value']}) : {sens}")

st.markdown(f"### {tr['interpretation']}")
if classe == 1:
    st.markdown(tr["explanation_high"])
else:
    st.markdown(tr["explanation_low"])
for phrase in explications:
    st.markdown(phrase)

if classe == 1:
    st.markdown(tr["conclusion_high"])
else:
    st.markdown(tr["conclusion_low"])

# -------------------------------
# Rapport IA Cohere
# -------------------------------

prompt = tr["prompt_template"].format(
    age=age,
    income=income,
    debt_ratio=debt_ratio,
    revolving=revolving,
    open_credit=open_credit,
    real_estate=real_estate,
    dependents=dependents,
    late_30=late_30,
    late_60=late_60,
    late_90=late_90,
    proba=proba
)

if "texte_ia" not in st.session_state:
    st.session_state["texte_ia"] = None

if st.button(tr["generate_ai_report"]):
    if "COHERE_API_KEY" not in st.secrets:
        st.error(tr["missing_api_key"])
        st.stop()
    with st.spinner(tr["ai_in_progress"]):
        co = cohere.Client(st.secrets["COHERE_API_KEY"])
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=300,
            temperature=0.3
        )
        st.session_state["texte_ia"] = response.generations[0].text.strip()

if st.session_state["texte_ia"]:
    texte_ia = st.session_state["texte_ia"]
    st.markdown(f'<div class="card">{texte_ia}</div>', unsafe_allow_html=True)

    # Génération PDF
    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, remove_accents(tr["pdf_title"]), ln=True, align="C")
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 10)
            self.cell(0, 10, f"{tr['page']} {self.page_no()}", align="C")

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 12)
    for line in texte_ia.split('\n'):
        clean_line = remove_accents(line.strip())
        if clean_line:
            pdf.multi_cell(180, 10, clean_line)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    st.download_button(
        label=tr["pdf_button"],
        data=pdf_output,
        file_name="rapport_credit_risquescore.pdf",
        mime="application/pdf"
    )

    # Génération Excel
    df_export = pd.DataFrame([{
        tr["age"]: age,
        tr["income"]: income,
        tr["debt_ratio"]: debt_ratio,
        tr["revolving"]: revolving,
        tr["open_credit"]: open_credit,
        tr["real_estate"]: real_estate,
        tr["dependents"]: dependents,
        tr["late_30"]: late_30,
        tr["late_60"]: late_60,
        tr["late_90"]: late_90,
        tr["default_prob"]: proba,
        tr["recommendation"]: tr["acceptance"] if classe == 0 else tr["rejection"]
    }])
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name="Résultat")
    excel_buffer.seek(0)

    st.download_button(
        label=tr["excel_button"],
        data=excel_buffer,
        file_name="resultat_credit_risquescore.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
