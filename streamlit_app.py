"""
Yeti-Travel Churn Prediction — Interactive Demo
=================================================
Streamlit app that loads pre-trained model artifacts and lets users
input school features for a live churn prediction with risk-tier.

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle, gzip, os, warnings
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────
SEED = 42
_dir = os.path.dirname(os.path.abspath(__file__))
# Find model artifacts
for _folder in ["outputs", "Outputs", os.path.join("mnt", "Machine Learning", "outputs")]:
    _path = os.path.join(_dir, _folder, "model_artifacts.pkl.gz")
    if os.path.exists(_path):
        MODEL_PATH = _path
        break
else:
    MODEL_PATH = os.path.join(_dir, "model_artifacts.pkl.gz")


# ───────────────────────────────────────────
# LOAD PRE-TRAINED MODEL (fast — just unpickle)
# ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the Stacked Ensemble...")
def load_model():
    with gzip.open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts


# ───────────────────────────────────────────
# PREDICTION HELPERS
# ───────────────────────────────────────────
def predict_single(row_df, preprocessor, base_models, meta):
    Xt = preprocessor.transform(row_df)
    base_preds = np.column_stack([
        mdl.predict_proba(Xt)[:, 1] for mdl in base_models.values()
    ])
    prob = meta.predict_proba(base_preds)[:, 1][0]
    return float(prob)


def risk_tier(prob, thr):
    if prob < 0.30:
        return "High Risk", "#FF4B4B", "Churn imminent — immediate outreach needed"
    elif prob < 0.50:
        return "At Risk", "#FFA726", "Vulnerable — proactive nurturing recommended"
    elif prob < 0.70:
        return "Likely Retain", "#66BB6A", "Stable — standard engagement"
    else:
        return "Strong Retain", "#00C853", "Loyal — ambassador & referral candidate"


def make_gauge(prob, tier_label, tier_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 48, 'color': 'white'}},
        title={'text': f"<b>{tier_label}</b>",
               'font': {'size': 22, 'color': tier_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2,
                     'tickcolor': '#555', 'tickfont': {'color': '#aaa'}},
            'bar': {'color': tier_color, 'thickness': 0.3},
            'bgcolor': '#1E1E2E',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30],  'color': 'rgba(255,75,75,0.15)'},
                {'range': [30, 50], 'color': 'rgba(255,167,38,0.15)'},
                {'range': [50, 70], 'color': 'rgba(102,187,106,0.15)'},
                {'range': [70, 100],'color': 'rgba(0,200,83,0.15)'},
            ],
            'threshold': {
                'line': {'color': '#D4A843', 'width': 3},
                'thickness': 0.8, 'value': 30
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
        height=320, margin=dict(t=80, b=20, l=40, r=40),
        font={'color': '#ccc'}
    )
    return fig


# ───────────────────────────────────────────
# STREAMLIT UI
# ───────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Yeti-Travel Churn Predictor",
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; }
    h1 { color: #D4A843 !important; }
    h2, h3 { color: #00C9A7 !important; }
    .metric-card {
        background: linear-gradient(135deg, #162040 0%, #1B2A4A 100%);
        border-radius: 12px; padding: 20px; text-align: center;
        border-left: 4px solid #D4A843;
    }
    .risk-badge {
        display: inline-block; padding: 8px 20px; border-radius: 20px;
        font-weight: 700; font-size: 1.1em; letter-spacing: 1px;
    }
    div[data-testid="stSidebar"] { background-color: #111B3C; }
    div[data-testid="stSidebar"] label { color: #E8E6E1 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Load model
    a = load_model()
    preprocessor = a['preprocessor']
    base_models = a['base_models']
    meta = a['meta']
    best_thr = a['best_thr']
    num_cols = a['num_cols']
    cat_cols = a['cat_cols']
    s = a['train_stats']
    cat_values = a['cat_values']

    # Header
    st.markdown("# 🏔️ Yeti-Travel — Churn Prediction Engine")
    st.markdown(
        "<p style='color:#8B95A8; font-size:1.1em; margin-top:-10px;'>"
        "Powered by a Stacked Ensemble (Logistic Regression + Random Forest + LightGBM + XGBoost) "
        "with 5-Fold Out-of-Fold predictions &nbsp;|&nbsp; <b style='color:#D4A843;'>GNRL Consulting</b></p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # Sidebar
    st.sidebar.markdown("## 📋 School Profile")
    st.sidebar.markdown("<p style='color:#8B95A8; font-size:0.85em;'>"
                        "Adjust features below to simulate a school booking</p>",
                        unsafe_allow_html=True)

    input_data = {}

    st.sidebar.markdown("### 🎯 Trip Details")
    input_data['Days'] = st.sidebar.slider("Trip Duration (Days)", 1, 30, int(s['Days']['median']))
    input_data['FPP'] = st.sidebar.slider("Full-Pay Passengers (FPP)", 1, 100, int(s['FPP']['median']))
    input_data['Total_Pax'] = st.sidebar.slider("Total Passengers", 1, 120, int(s['Total_Pax']['median']))
    input_data['Tuition'] = st.sidebar.slider("Tuition ($)", 0, 5000, int(s['Tuition']['median']), step=50)
    input_data['Grade_Range'] = st.sidebar.slider("Grade Range", 0, 12, int(s['Grade_Range']['median']),
                                                   help="Top SHAP driver — broader coverage → retention")
    input_data['From_Grade'] = st.sidebar.slider("From Grade", 0, 12, 5)
    input_data['To_Grade'] = input_data['From_Grade'] + input_data['Grade_Range']

    st.sidebar.markdown("### ⏰ Planning & Timing")
    input_data['LeadTime_Days'] = st.sidebar.slider("Lead Time (Days)", 30, 700, int(s['LeadTime_Days']['median']))
    input_data['CampaignWindow_Days'] = st.sidebar.slider("Campaign Window (Days)", 0, 400,
                                                           int(s.get('CampaignWindow_Days', {}).get('median', 100)))
    input_data['EarlyPlanner_Flag'] = st.sidebar.selectbox("Early Planner?", [0, 1], index=0)

    st.sidebar.markdown("### 🤝 Engagement")
    input_data['NumberOfMeetingswithParents'] = st.sidebar.slider("Parent Meetings", 0, 10,
                                                                   int(s.get('NumberOfMeetingswithParents', {}).get('median', 1)))
    input_data['Parent_Meeting_Flag'] = st.sidebar.selectbox("Had Parent Meeting?", [0, 1], index=1)
    input_data['HadFirstMeeting_Flag'] = st.sidebar.selectbox("Had First Meeting?", [0, 1], index=1)
    input_data['CareIntensity_Score'] = st.sidebar.slider("Care Intensity Score", 0.0, 8.0,
                                                           float(s.get('CareIntensity_Score', {}).get('median', 2.0)), step=0.5)
    input_data['School_Sponsor'] = st.sidebar.selectbox("School Sponsor", [0, 1], index=1)

    st.sidebar.markdown("### 💰 Pricing & Risk")
    input_data['Cancel_Ratio'] = st.sidebar.slider("Cancel Ratio", 0.0, 1.0,
                                                     float(s.get('Cancel_Ratio', {}).get('median', 0.05)), step=0.01,
                                                     help="High = churn risk")
    input_data['Discount_Ratio'] = st.sidebar.slider("Discount Ratio", 0.0, 1.0,
                                                       float(s.get('Discount_Ratio', {}).get('median', 0.1)), step=0.01)
    input_data['Insurance_Penetration'] = st.sidebar.slider("Insurance Penetration", 0.0, 1.0,
                                                              float(s.get('Insurance_Penetration', {}).get('median', 0.0)), step=0.01)

    st.sidebar.markdown("### 🏫 School Profile")
    input_data['Total_School_Enrollment'] = st.sidebar.slider("School Enrollment", 50, 3000,
                                                               int(s.get('Total_School_Enrollment', {}).get('median', 500)))

    default_cat_map = {
        'Travel_Type': 'Domestic', 'SPR_Product_Type': 'School', 'Region': 'Other',
        'School_Type': 'Public', 'Poverty_Code': 'C', 'Income_Level': 'I', 'SchoolSizeIndicator': 'M',
    }
    for c in cat_cols:
        if c in cat_values and len(cat_values[c]) > 0:
            default = default_cat_map.get(c, cat_values[c][0])
            if default not in cat_values[c]:
                default = cat_values[c][0]
            input_data[c] = st.sidebar.selectbox(c.replace('_', ' '), cat_values[c],
                                                  index=cat_values[c].index(default))

    # Fill remaining with median
    for c in num_cols:
        if c not in input_data:
            input_data[c] = s.get(c, {}).get('median', 0.0)

    # Derived features
    input_data['Tuition_per_Day'] = input_data['Tuition'] / max(input_data['Days'], 1)
    input_data['Revenue_per_FPP'] = input_data.get('SPR_Group_Revenue', s.get('SPR_Group_Revenue', {}).get('median', 0)) / max(input_data['FPP'], 1)
    input_data['Tuition_per_FPP'] = input_data['Tuition'] / max(input_data['FPP'], 1)
    input_data['Group_Penetration'] = input_data['Total_Pax'] / max(input_data['Total_School_Enrollment'], 1)
    input_data['MeetingsPerMonth'] = input_data['NumberOfMeetingswithParents'] / max(input_data['LeadTime_Days'] / 30.0, 0.1)
    input_data['NonFPP_Share'] = input_data.get('Num_of_Non_FPP_PAX', 0) / max(input_data['Total_Pax'], 1)
    input_data['Logistics_Load'] = input_data['Days'] * input_data['Total_Pax']

    # Build row
    row_df = pd.DataFrame([input_data])
    for c in cat_cols:
        if c in row_df.columns:
            row_df[c] = row_df[c].astype(str).fillna('MISSING')
    for c in num_cols + cat_cols:
        if c not in row_df.columns:
            row_df[c] = np.nan
    row_df = row_df[num_cols + cat_cols]

    # Predict
    prob = predict_single(row_df, preprocessor, base_models, meta)
    tier, color, desc = risk_tier(prob, best_thr)
    predicted_label = "Retained" if prob >= best_thr else "Churned"

    # Main content
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("## 🔮 Prediction Result")
        fig = make_gauge(prob, tier, color)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;'>"
            f"<span class='risk-badge' style='background:{color}22; color:{color}; "
            f"border: 2px solid {color};'>{tier.upper()}</span>"
            f"<p style='color:#8B95A8; margin-top:10px;'>{desc}</p>"
            f"</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("## 📊 Key Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <p style="color:#8B95A8; margin:0; font-size:0.85em;">RETENTION PROBABILITY</p>
                <p style="color:#D4A843; font-size:2.2em; font-weight:700; margin:5px 0;">{prob:.1%}</p>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color};">
                <p style="color:#8B95A8; margin:0; font-size:0.85em;">CHURN PROBABILITY</p>
                <p style="color:#FF6B6B; font-size:2.2em; font-weight:700; margin:5px 0;">{1-prob:.1%}</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#00C9A7;">
                <p style="color:#8B95A8; margin:0; font-size:0.85em;">PREDICTION</p>
                <p style="color:{'#00C9A7' if predicted_label=='Retained' else '#FF6B6B'}; font-size:1.6em; font-weight:700; margin:5px 0;">{predicted_label}</p>
                <p style="color:#555; font-size:0.75em; margin:0;">Threshold: {best_thr:.2f}</p>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:#00C9A7;">
                <p style="color:#8B95A8; margin:0; font-size:0.85em;">MODEL</p>
                <p style="color:#D4A843; font-size:1.1em; font-weight:700; margin:5px 0;">Stacked Ensemble</p>
                <p style="color:#555; font-size:0.75em; margin:0;">AUC 0.851 · F1 0.833</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🤖 Base Model Breakdown")
        Xt = preprocessor.transform(row_df)
        model_colors = {'LogReg': '#D4A843', 'RandomForest': '#00C9A7',
                        'LightGBM': '#66BB6A', 'XGBoost': '#FFA726'}
        for name, mdl in base_models.items():
            p = mdl.predict_proba(Xt)[:, 1][0]
            st.markdown(f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; color:#E8E6E1; font-size:0.9em;">
                    <span>{name}</span><span style="color:{model_colors[name]};">{p:.1%}</span>
                </div>
                <div style="background:#1B2A4A; border-radius:6px; height:8px; overflow:hidden;">
                    <div style="width:{p*100:.1f}%; height:100%; background:{model_colors[name]};
                                border-radius:6px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Top Churn Drivers (from SHAP Analysis)")
    d1, d2, d3, d4, d5 = st.columns(5)
    drivers = [
        (d1, "🎓", "Grade Range", "Broader coverage → retention", "#00C9A7"),
        (d2, "👥", "FPP / Enrollment", "Higher penetration → loyalty", "#00C9A7"),
        (d3, "⏱️", "Lead Time", "Early planners return", "#D4A843"),
        (d4, "❌", "Cancel Ratio", "High cancellations → churn", "#FF6B6B"),
        (d5, "💲", "Revenue / FPP", "Price sensitivity signal", "#FFA726"),
    ]
    for col, icon, name, desc, clr in drivers:
        with col:
            st.markdown(f"""
            <div style="background:#162040; border-radius:10px; padding:15px; text-align:center;
                        border-top: 3px solid {clr}; height:140px;">
                <p style="font-size:1.8em; margin:0;">{icon}</p>
                <p style="color:{clr}; font-weight:700; margin:5px 0; font-size:0.95em;">{name}</p>
                <p style="color:#8B95A8; font-size:0.78em; margin:0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>")
    st.markdown(
        "<p style='text-align:center; color:#555; font-size:0.8em;'>"
        "GNRL Consulting · Polimi — Business Analytics & Data Science · "
        "Yeti-Travel Churn Prediction Engine</p>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()

