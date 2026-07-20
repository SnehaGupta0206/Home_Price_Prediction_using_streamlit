import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="House Price Predictor", page_icon="", layout="wide")

model = joblib.load('models/best_model.pkl')
model_columns = joblib.load('models/model_columns.pkl')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp {
    background-color: #14213D;
    background-image:
        linear-gradient(rgba(232,234,240,0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,234,240,0.045) 1px, transparent 1px);
    background-size: 32px 32px;
    color: #E8EAF0;
    font-family: 'Inter', sans-serif;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #1B2A47;
    border-right: 1px solid #2E3E5F;
}
section[data-testid="stSidebar"] .block-container { padding-top: 28px; }

.sidebar-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: #F5F2EA;
    margin-bottom: 2px;
}
.sidebar-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #8D99AE;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px dashed #3A4A6B;
    padding-bottom: 14px;
    margin-bottom: 18px;
}
.sidebar-section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #E9C46A;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 18px 0 8px 0;
}

label, .stSlider label, .stNumberInput label, .stSelectbox label {
    font-family: 'Inter', sans-serif !important;
    color: #C7CEDB !important;
    font-size: 0.82rem !important;
}
.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ---------- Sidebar button ---------- */
div.stButton > button {
    background: linear-gradient(135deg, #E9C46A, #D9AE4E);
    color: #14213D;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 4px;
    padding: 13px 0;
    width: 100px;
    letter-spacing: 0.6px;
    font-size: 0.9rem;
    box-shadow: 0 4px 14px rgba(233,196,106,0.18);
    transition: all 0.18s ease;
    margin-top: 10px;
}
div.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(233,196,106,0.35);
    transform: translateY(-1px);
    color: #14213D;
}

/* ---------- Main hero ---------- */
.hero-row { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
.hero-title {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: 2.4rem;
    color: #F5F2EA;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #8D99AE;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-bottom: 1px dashed #3A4A6B;
    padding-bottom: 16px;
    margin-bottom: 28px;
    margin-top: 4px;
}

/* ---------- Empty state ---------- */
.empty-state {
    border: 1px dashed #2E3E5F;
    border-radius: 6px;
    padding: 60px 40px;
    text-align: center;
    color: #6B7A99;
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
}
.empty-state span {
    display: block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #E9C46A;
    margin-bottom: 10px;
}

/* ---------- Result spec card ---------- */
.spec-card {
    border: 1px dashed #E9C46A;
    background: linear-gradient(135deg, rgba(233,196,106,0.08), rgba(233,196,106,0.02));
    border-radius: 6px;
    padding: 30px 32px;
    margin-bottom: 8px;
}
.spec-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #E9C46A;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.spec-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    color: #F5F2EA;
}
.spec-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #8D99AE;
    margin-top: 12px;
    max-width: 560px;
}

/* ---------- Chart section ---------- */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #E9C46A;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 30px 0 4px 0;
}
.section-desc {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #8D99AE;
    margin-bottom: 14px;
}

hr { border-color: #2E3E5F; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title"> Price Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Ames, Iowa Housing Model</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section-label"> Structure</div>', unsafe_allow_html=True)
    overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    first_flr_sf = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000)
    second_flr_sf = st.number_input("2nd Floor Area (sq ft)", 0, 2500, 0)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)

    st.markdown('<div class="sidebar-section-label"> History &amp; Amenities</div>', unsafe_allow_html=True)
    year_built = st.number_input("Year Built", 1870, 2026, 2005)
    year_remod = st.number_input("Year Remodeled", 1870, 2026, 2005)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    half_bath = st.slider("Half Bathrooms", 0, 2, 0)
    fireplaces = st.slider("Fireplaces", 0, 3, 0)
    neighborhood = st.selectbox(
        "Neighborhood",
        ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NridgHt"]
    )

    predict_clicked = st.button("Predict Price")

st.markdown("""
<div class="hero-row">
<svg width="40" height="40" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
  <path d="M4 22L22 6L40 22" stroke="#E9C46A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M9 19V38H35V19" stroke="#E9C46A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M18 38V27H26V38" stroke="#E9C46A" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M27 8V13" stroke="#E9C46A" stroke-width="2" stroke-linecap="round"/>
</svg>
<div class="hero-title">House Price Predictor</div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="hero-sub">Advanced Regression Model · Stacked Ensemble (Ridge + Lasso + XGBoost)</div>',
    unsafe_allow_html=True
)

def build_input(yr_built=None, yr_remod=None):
    yr_built = year_built if yr_built is None else yr_built
    yr_remod = year_remod if yr_remod is None else yr_remod

    input_dict = {col: 0 for col in model_columns}

    total_sf = total_bsmt_sf + first_flr_sf + second_flr_sf
    total_bath = full_bath + (0.5 * half_bath)
    house_age = 2026 - yr_built
    years_since_remodel = 2026 - yr_remod

    input_dict['OverallQual'] = overall_qual
    input_dict['GrLivArea'] = np.log1p(gr_liv_area)
    input_dict['TotalBsmtSF'] = np.log1p(total_bsmt_sf) if total_bsmt_sf > 0 else 0
    input_dict['1stFlrSF'] = np.log1p(first_flr_sf)
    input_dict['2ndFlrSF'] = second_flr_sf
    input_dict['GarageCars'] = garage_cars
    input_dict['TotalSF'] = np.log1p(total_sf)
    input_dict['TotalBath'] = total_bath
    input_dict['HouseAge'] = house_age
    input_dict['YearsSinceRemodel'] = years_since_remodel
    input_dict['YearBuilt'] = yr_built
    input_dict['YearRemodAdd'] = yr_remod
    input_dict['Fireplaces'] = fireplaces
    input_dict['HasFireplace'] = 1 if fireplaces > 0 else 0
    input_dict['HasGarage'] = 1 if garage_cars > 0 else 0
    input_dict['HasBsmt'] = 1 if total_bsmt_sf > 0 else 0

    neighborhood_col = f'Neighborhood_{neighborhood}'
    if neighborhood_col in input_dict:
        input_dict[neighborhood_col] = 1

    return pd.DataFrame([input_dict])[model_columns]

if not predict_clicked:
    st.markdown("""
    <div class="empty-state">
        <span>Awaiting Input</span>
        Fill in the house details in the sidebar and click <b>Predict Price</b> to see the
        estimated sale price and the build-year / renovation impact chart here.
    </div>
    """, unsafe_allow_html=True)
else:
    input_df = build_input()
    pred_log = model.predict(input_df)[0]
    pred_price = np.expm1(pred_log)

    st.markdown(f"""
    <div class="spec-card">
        <div class="spec-label">Estimated Sale Price</div>
        <div class="spec-value">${pred_price:,.0f}</div>
        <div class="spec-note">
            Based on a stacked ensemble (Ridge + Lasso + XGBoost) trained on the
            Ames, Iowa housing dataset (2006–2010). For reference only — not a formal appraisal.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label"> Build Year vs. Renovation Impact</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Predicted price for this exact house across a range of build years — '
        'comparing a version that was remodeled at your selected year vs. one that was never remodeled.</div>',
        unsafe_allow_html=True
    )

    build_years = list(range(1900, 2027, 5))
    renovated_prices = []
    never_renovated_prices = []

    remod_offset = year_remod - year_built  # keep the same "years after build" gap

    for by in build_years:
        remod_year = min(by + max(remod_offset, 0), 2026)
        df_reno = build_input(yr_built=by, yr_remod=remod_year)
        renovated_prices.append(np.expm1(model.predict(df_reno)[0]))

        df_never = build_input(yr_built=by, yr_remod=by)
        never_renovated_prices.append(np.expm1(model.predict(df_never)[0]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=build_years, y=renovated_prices,
        mode='lines+markers', name='Renovated',
        line=dict(color='#E9C46A', width=3),
        marker=dict(size=6, color='#E9C46A')
    ))
    fig.add_trace(go.Scatter(
        x=build_years, y=never_renovated_prices,
        mode='lines+markers', name='Never Renovated',
        line=dict(color='#8D99AE', width=2, dash='dash'),
        marker=dict(size=6, color='#8D99AE')
    ))
    fig.add_vline(x=year_built, line_width=1, line_dash="dot", line_color="rgba(233,196,106,0.55)",
                  annotation_text="Your build year", annotation_font_color="#E9C46A")

    fig.update_layout(
        plot_bgcolor='#1F2E4D',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='JetBrains Mono, monospace', color='#C7CEDB', size=12),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0,
                    bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(title='Build Year', gridcolor='#2E3E5F', zeroline=False),
        yaxis=dict(title='Predicted Price ($)', gridcolor='#2E3E5F', zeroline=False, tickprefix='$'),
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        '<div class="section-desc">The gap between the two lines shows the price premium the model '
        'attributes to renovation at each build era — a wider gap means renovation matters more for houses of that age.</div>',
        unsafe_allow_html=True
    )