"""
Yeti-Travel Churn Prediction — Interactive Demo
=================================================
Streamlit app that lets users input school features and get a live
churn probability with risk-tier classification.

Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import os, warnings
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────
SEED = 42
# Auto-detect data path: works from project root or from the Machine Learning folder
_dir = os.path.dirname(os.path.abspath(__file__))
_candidate1 = os.path.join(_dir, "outputs", "Yeti_Travel_Cleaned_Merged.csv")
_candidate2 = os.path.join(_dir, "mnt", "Machine Learning", "outputs", "Yeti_Travel_Cleaned_Merged.csv")
DATA_PATH = _candidate1 if os.path.exists(_candidate1) else _candidate2

# Feature lists (must match notebook exactly)
NUMERIC_FEATURES = [
    'From_Grade', 'To_Grade', 'Days', 'Cancelled_Pax', 'Total_Discount_Pax',
    'FPP', 'Total_Pax', 'FPP_to_School_enrollment',
    'Tuition', 'FRP_Active', 'FRP_Cancelled', 'FRP_Take_up_percent_',
    'EZ_Pay_Take_Up_Rate', 'SPR_Group_Revenue', 'FPP_to_PAX', 'Num_of_Non_FPP_PAX',
    'CRM_Segment', 'MDR_High_Grade',
    'Total_School_Enrollment', 'NumberOfMeetingswithParents',
    'DifferenceTraveltoFirstMeeting', 'DifferenceTraveltoLastMeeting',
    'Poverty_Severity', 'Income_Ordinal',
    'School_Sponsor', 'Parent_Meeting_Flag', 'SPR_New_Existing',
    'LeadTime_Days', 'CampaignWindow_Days', 'EarlyComm_LeadDays', 'LateComm_LeadDays',
    'EarlyPlanner_Flag',
    'MeetingsPerMonth', 'HadFirstMeeting_Flag',
    'Group_Penetration', 'Grade_Range',
    'Tuition_per_Day', 'Revenue_per_FPP', 'Tuition_per_FPP',
    'Discount_Ratio', 'Cancel_Ratio', 'Insurance_Penetration', 'Insurance_LossRate',
    'Tuition_vs_PovertyMed', 'PriceSensitivity_Idx',
    'NonFPP_Share', 'Logistics_Load', 'CareIntensity_Score',
]

CAT_FEATURES = [
    'Program_Code', 'Group_State', 'Travel_Type', 'SPR_Product_Type',
    'DepartureMonth', 'GroupGradeTypeLow', 'GroupGradeTypeHigh', 'GroupGradeType',
    'MajorProgramCode', 'Special_Pay', 'Poverty_Code', 'Region', 'School_Type',
    'Income_Level', 'SchoolGradeTypeLow', 'SchoolGradeTypeHigh', 'SchoolGradeType',
    'SchoolSizeIndicator', 'MDR_Low_Grade',
]


# ───────────────────────────────────────────
# TRAIN & CACHE MODEL
# ───────────────────────────────────────────
@st.cache_resource(show_spinner="Training the Stacked Ensemble — this only happens once...")
def load_and_train():
    """Load data, build preprocessor, train stacked ensemble, return artifacts."""
    df = pd.read_csv(DATA_PATH)
    train_df = df[df['Split'] == 'model'].copy()
    test_df  = df[df['Split'] == 'test'].copy()

    num_cols = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    cat_cols = [c for c in CAT_FEATURES if c in train_df.columns]

    # Force numeric columns to numeric (coerce strings to NaN)
    for c in num_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors='coerce')

    X_train_raw = train_df[num_cols + cat_cols].copy()
    y_train     = train_df['Retained'].astype(int).values

    for c in cat_cols:
        X_train_raw[c] = X_train_raw[c].astype(str).fillna('MISSING')

    # Preprocessor
    num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler())])
    cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore',
                                               min_frequency=5, sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols),
    ], remainder='drop', verbose_feature_names_out=False)

    Xt_train = preprocessor.fit_transform(X_train_raw)

    # Base models
    base_models = {
        'LogReg': LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced',
                                      random_state=SEED, n_jobs=-1),
        'RandomForest': RandomForestClassifier(n_estimators=500, max_depth=None,
                                                min_samples_leaf=2,
                                                class_weight='balanced_subsample',
                                                random_state=SEED, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03,
                                         num_leaves=63, min_child_samples=20,
                                         subsample=0.9, colsample_bytree=0.9,
                                         reg_alpha=0.1, reg_lambda=0.1,
                                         class_weight='balanced',
                                         random_state=SEED, n_jobs=-1, verbose=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=800, learning_rate=0.03,
                                       max_depth=6, subsample=0.9,
                                       colsample_bytree=0.9, reg_alpha=0.1,
                                       reg_lambda=1.0, eval_metric='logloss',
                                       random_state=SEED, n_jobs=-1,
                                       tree_method='hist'),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = pd.DataFrame(index=np.arange(len(y_train)),
                              columns=list(base_models.keys()), dtype=float)

    for name, mdl in base_models.items():
        probs = cross_val_predict(mdl, Xt_train, y_train, cv=skf,
                                   method='predict_proba', n_jobs=-1)[:, 1]
        oof_preds[name] = probs
        mdl.fit(Xt_train, y_train)  # refit on full data

    # Meta-learner
    meta = LogisticRegression(max_iter=2000, random_state=SEED)
    meta.fit(oof_preds.values, y_train)

    # Optimal threshold
    stacked_oof = meta.predict_proba(oof_preds.values)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_train, (stacked_oof >= t).astype(int)) for t in thresholds]
    best_thr = float(thresholds[np.argmax(f1s)])

    # Collect training stats for sidebar
    train_stats = {}
    for c in num_cols:
        col = pd.to_numeric(train_df[c], errors='coerce').dropna()
        if len(col) == 0:
            train_stats[c] = {'min': 0, 'max': 1, 'mean': 0, 'median': 0, 'q25': 0, 'q75': 1}
        else:
            train_stats[c] = {
                'min': float(col.min()), 'max': float(col.max()),
                'mean': float(col.mean()), 'median': float(col.median()),
                'q25': float(col.quantile(0.25)), 'q75': float(col.quantile(0.75))
            }
    cat_values = {}
    for c in cat_cols:
        cat_values[c] = sorted(train_df[c].dropna().astype(str).unique().tolist())

    return (preprocessor, base_models, meta, best_thr,
            num_cols, cat_cols, train_stats, cat_values, train_df)


# ───────────────────────────────────────────
# PREDICTION HELPERS
# ───────────────────────────────────────────
def predict_single(row_df, preprocessor, base_models, meta):
    """Run full pipeline on a single-row DataFrame and return retention probability."""
    Xt = preprocessor.transform(row_df)
    base_preds = np.column_stack([
        mdl.predict_proba(Xt)[:, 1] for mdl in base_models.values()
    ])
    prob = meta.predict_proba(base_preds)[:, 1][0]
    return float(prob)


def risk_tier(prob, thr):
    """Map retention probability to risk tier."""
    if prob < 0.30:
        return "High Risk", "#FF4B4B", "Churn imminent — immediate outreach needed"
    elif prob < 0.50:
        return "At Risk", "#FFA726", "Vulnerable — proactive nurturing recommended"
    elif prob < 0.70:
        return "Likely Retain", "#66BB6A", "Stable — standard engagement"
    else:
        return "Strong Retain", "#00C853", "Loyal — ambassador & referral candidate"


def make_gauge(prob, tier_label, tier_color):
    """Build a Plotly gauge showing retention probability."""
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
                'thickness': 0.8, 'value': 30  # optimal threshold marker
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

    # Custom CSS
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

    # ── Load model ──
    (preprocessor, base_models, meta, best_thr,
     num_cols, cat_cols, train_stats, cat_values, train_df) = load_and_train()

    # ── Header ──
    st.markdown("# 🏔️ Yeti-Travel — Churn Prediction Engine")
    st.markdown(
        "<p style='color:#8B95A8; font-size:1.1em; margin-top:-10px;'>"
        "Powered by a Stacked Ensemble (Logistic Regression + Random Forest + LightGBM + XGBoost) "
        "with 5-Fold Out-of-Fold predictions &nbsp;|&nbsp; <b style='color:#D4A843;'>GNRL Consulting</b></p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ── Sidebar: Input Features ──
    st.sidebar.markdown("## 📋 School Profile")
    st.sidebar.markdown("<p style='color:#8B95A8; font-size:0.85em;'>"
                        "Adjust features below to simulate a school booking</p>",
                        unsafe_allow_html=True)

    input_data = {}

    # --- Key Numeric Sliders ---
    st.sidebar.markdown("### 🎯 Trip Details")

    s = train_stats
    input_data['Days'] = st.sidebar.slider(
        "Trip Duration (Days)", 1, 30,
        int(s['Days']['median']), help="Number of travel days")
    input_data['FPP'] = st.sidebar.slider(
        "Full-Pay Passengers (FPP)", 1, 100,
        int(s['FPP']['median']), help="Number of full-pay passengers")
    input_data['Total_Pax'] = st.sidebar.slider(
        "Total Passengers", 1, 120,
        int(s['Total_Pax']['median']), help="Total passengers including non-FPP")
    input_data['Tuition'] = st.sidebar.slider(
        "Tuition ($)", 0, 5000,
        int(s['Tuition']['median']), step=50, help="Trip tuition amount")
    input_data['Grade_Range'] = st.sidebar.slider(
        "Grade Range", 0, 12,
        int(s['Grade_Range']['median']),
        help="Difference between highest and lowest grade (top SHAP driver)")
    input_data['From_Grade'] = st.sidebar.slider("From Grade", 0, 12, 5)
    input_data['To_Grade'] = input_data['From_Grade'] + input_data['Grade_Range']

    st.sidebar.markdown("### ⏰ Planning & Timing")
    input_data['LeadTime_Days'] = st.sidebar.slider(
        "Lead Time (Days)", 30, 700,
        int(s['LeadTime_Days']['median']),
        help="Days between first contact and departure")
    input_data['CampaignWindow_Days'] = st.sidebar.slider(
        "Campaign Window (Days)", 0, 400,
        int(s.get('CampaignWindow_Days', {}).get('median', 100)))
    input_data['EarlyPlanner_Flag'] = st.sidebar.selectbox(
        "Early Planner?", [0, 1],
        index=0, help="1 if lead time > 75th percentile")

    st.sidebar.markdown("### 🤝 Engagement")
    input_data['NumberOfMeetingswithParents'] = st.sidebar.slider(
        "Parent Meetings", 0, 10,
        int(s.get('NumberOfMeetingswithParents', {}).get('median', 1)))
    input_data['Parent_Meeting_Flag'] = st.sidebar.selectbox(
        "Had Parent Meeting?", [0, 1], index=1)
    input_data['HadFirstMeeting_Flag'] = st.sidebar.selectbox(
        "Had First Meeting?", [0, 1], index=1)
    input_data['CareIntensity_Score'] = st.sidebar.slider(
        "Care Intensity Score", 0.0, 8.0,
        float(s.get('CareIntensity_Score', {}).get('median', 2.0)), step=0.5,
        help="Composite engagement score")
    input_data['School_Sponsor'] = st.sidebar.selectbox(
        "School Sponsor", [0, 1], index=1)

    st.sidebar.markdown("### 💰 Pricing & Risk")
    input_data['Cancel_Ratio'] = st.sidebar.slider(
        "Cancel Ratio", 0.0, 1.0,
        float(s.get('Cancel_Ratio', {}).get('median', 0.05)), step=0.01,
        help="Cancelled pax / total pax (high = churn risk)")
    input_data['Discount_Ratio'] = st.sidebar.slider(
        "Discount Ratio", 0.0, 1.0,
        float(s.get('Discount_Ratio', {}).get('median', 0.1)), step=0.01)
    input_data['Insurance_Penetration'] = st.sidebar.slider(
        "Insurance Penetration", 0.0, 1.0,
        float(s.get('Insurance_Penetration', {}).get('median', 0.0)), step=0.01)

    st.sidebar.markdown("### 🏫 School Profile")
    input_data['Total_School_Enrollment'] = st.sidebar.slider(
        "School Enrollment", 50, 3000,
        int(s.get('Total_School_Enrollment', {}).get('median', 500)))

    # Key categorical features
    default_cat_map = {
        'Travel_Type': 'Domestic',
        'SPR_Product_Type': 'School',
        'Region': 'Other',
        'School_Type': 'Public',
        'Poverty_Code': 'C',
        'Income_Level': 'I',
        'SchoolSizeIndicator': 'M',
    }
    for c in cat_cols:
        if c in cat_values and len(cat_values[c]) > 0:
            default = default_cat_map.get(c, cat_values[c][0])
            if default not in cat_values[c]:
                default = cat_values[c][0]
            input_data[c] = st.sidebar.selectbox(
                c.replace('_', ' '), cat_values[c],
                index=cat_values[c].index(default))

    # Fill remaining numeric features with median
    for c in num_cols:
        if c not in input_data:
            input_data[c] = s.get(c, {}).get('median', 0.0)

    # Compute derived features
    input_data['Tuition_per_Day'] = input_data['Tuition'] / max(input_data['Days'], 1)
    input_data['Revenue_per_FPP'] = input_data.get('SPR_Group_Revenue', s.get('SPR_Group_Revenue', {}).get('median', 0)) / max(input_data['FPP'], 1)
    input_data['Tuition_per_FPP'] = input_data['Tuition'] / max(input_data['FPP'], 1)
    input_data['Group_Penetration'] = input_data['Total_Pax'] / max(input_data['Total_School_Enrollment'], 1)
    input_data['MeetingsPerMonth'] = input_data['NumberOfMeetingswithParents'] / max(input_data['LeadTime_Days'] / 30.0, 0.1)
    input_data['NonFPP_Share'] = input_data.get('Num_of_Non_FPP_PAX', 0) / max(input_data['Total_Pax'], 1)
    input_data['Logistics_Load'] = input_data['Days'] * input_data['Total_Pax']

    # Build single-row DataFrame
    row_df = pd.DataFrame([input_data])
    for c in cat_cols:
        if c in row_df.columns:
            row_df[c] = row_df[c].astype(str).fillna('MISSING')
    # Ensure all expected columns exist
    for c in num_cols + cat_cols:
        if c not in row_df.columns:
            row_df[c] = np.nan
    row_df = row_df[num_cols + cat_cols]

    # ── Predict ──
    prob = predict_single(row_df, preprocessor, base_models, meta)
    tier, color, desc = risk_tier(prob, best_thr)
    predicted_label = "Retained" if prob >= best_thr else "Churned"

    # ── Main Content ──
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("## 🔮 Prediction Result")
        fig = make_gauge(prob, tier, color)
        st.plotly_chart(fig, use_container_width=True)

        # Risk badge
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
            churn_prob = 1 - prob
            st.markdown(f"""
            <div class="metric-card" style="border-left-color:{color};">
                <p style="color:#8B95A8; margin:0; font-size:0.85em;">CHURN PROBABILITY</p>
                <p style="color:#FF6B6B; font-size:2.2em; font-weight:700; margin:5px 0;">{churn_prob:.1%}</p>
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

        # Individual model probabilities
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🤖 Base Model Breakdown")

        Xt = preprocessor.transform(row_df)
        model_colors = {'LogReg': '#D4A843', 'RandomForest': '#00C9A7',
                        'LightGBM': '#66BB6A', 'XGBoost': '#FFA726'}
        for name, mdl in base_models.items():
            p = mdl.predict_proba(Xt)[:, 1][0]
            bar_html = f"""
            <div style="margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; color:#E8E6E1; font-size:0.9em;">
                    <span>{name}</span><span style="color:{model_colors[name]};">{p:.1%}</span>
                </div>
                <div style="background:#1B2A4A; border-radius:6px; height:8px; overflow:hidden;">
                    <div style="width:{p*100:.1f}%; height:100%; background:{model_colors[name]};
                                border-radius:6px;"></div>
                </div>
            </div>"""
            st.markdown(bar_html, unsafe_allow_html=True)

    # ── Bottom: Feature Importance Reminder ──
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

    # Footer
    st.markdown("<br>")
    st.markdown(
        "<p style='text-align:center; color:#555; font-size:0.8em;'>"
        "GNRL Consulting · Polimi — Business Analytics & Data Science · "
        "Yeti-Travel Churn Prediction Engine</p>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
