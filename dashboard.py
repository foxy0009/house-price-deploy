import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EstateIntel | AI Price Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM STYLING (CSS) ---
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #333;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("⚠️ Model files missing! Please ensure .joblib and .pkl files are in the directory.")
        return None, None, None

model, scaler, model_columns = load_artifacts()

# --- 4. HEADER SECTION ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🏢 EstateIntel")
    st.markdown("### Turkish Real Estate Value Estimator")
    st.markdown("Leveraging **XGBoost** machine learning to predict accurate property valuations.")

with col_head2:
    st.image("https://cdn-icons-png.flaticon.com/512/1040/1040993.png", width=100) # Placeholder Icon

st.divider()

# --- 5. INPUT SECTION (TABS) ---
col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("📝 Property Details")
    
    tab1, tab2, tab3 = st.tabs(["📍 Location", "📐 Dimensions", "🏗️ Building Info"])

    with tab1:
        city = st.selectbox("City", ["Istanbul", "Ankara", "Izmir", "Antalya", "Other"], index=0)
        heating = st.selectbox("Heating Type", ["Kombi Doğalgaz", "Merkezi Doğalgaz", "Other"], index=0)

    with tab2:
        col_dim1, col_dim2 = st.columns(2)
        with col_dim1:
            net_sqm = st.number_input("Net Area (m²)", min_value=10, max_value=500, value=100, step=5)
        with col_dim2:
            rooms = st.number_input("Rooms", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
        
        # Bathroom is visual only as per your logic
        bathrooms = st.slider("Bathrooms", 1, 5, 1)

    with tab3:
        age = st.slider("Building Age", 0, 50, 5)
        col_bld1, col_bld2 = st.columns(2)
        with col_bld1:
            total_floors = st.number_input("Total Floors", min_value=1, value=5)
        with col_bld2:
            floor = st.number_input("Flat Floor", min_value=-2, value=2)

    st.markdown("---")
    predict_btn = st.button("✨ Analyze & Estimate Value", type="primary")

# --- 6. PREDICTION LOGIC ---
def predict_price():
    if model is None:
        return 0
    
    input_data = {
        "Net_Metrekare": net_sqm,
        "Oda_Sayısı": rooms,
        "Bulunduğu_Kat": floor,
        "Binanın_Yaşı": age,
        "Binanın_Kat_Sayısı": total_floors,
    }
    
    df = pd.DataFrame([input_data])
    
    # One-Hot Encoding
    sehir_col = f"Şehir_{city.lower()}" if city != "Other" else "other"
    heat_col = f"Isıtma_Tipi_{heating}" if heating != "Other" else "other"
    
    df = df.reindex(columns=model_columns, fill_value=0)
    
    if sehir_col in df.columns: df[sehir_col] = 1
    if heat_col in df.columns: df[heat_col] = 1
    
    # Scale
    scale_cols = ['Net_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 
                  'Binanın_Yaşı', 'Binanın_Kat_Sayısı']
    
    # Robust handling if scaler expects cols that aren't here
    try:
        df[scale_cols] = scaler.transform(df[scale_cols])
    except Exception as e:
        st.warning(f"Scaling issue: {e}")

    log_pred = model.predict(df)[0]
    price = np.expm1(log_pred)
    return price

# --- 7. RESULTS SECTION ---
with col_results:
    st.subheader("📊 Valuation Report")
    
    if predict_btn:
        with st.spinner("Crunching market data..."):
            time.sleep(0.5) # UX: Artificial delay to make it feel like "thinking"
            estimated_price = predict_price()
        
        # Metrics Row
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Estimated Value", value=f"{estimated_price:,.0f} TL")
        with m_col2:
            price_per_sqm = estimated_price / net_sqm
            st.metric(label="Price per m²", value=f"{price_per_sqm:,.0f} TL")

        # Plotly Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = estimated_price,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Confidence Range"},
            gauge = {
                'axis': {'range': [estimated_price*0.5, estimated_price*1.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#00cc96"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [estimated_price*0.5, estimated_price*0.8], 'color': '#ff6961'},
                    {'range': [estimated_price*0.8, estimated_price*1.2], 'color': '#fdfd96'},
                    {'range': [estimated_price*1.2, estimated_price*1.5], 'color': '#ff6961'}],
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Recap
        with st.expander("See Input Summary"):
            st.dataframe(pd.DataFrame({
                "Feature": ["City", "Net Area", "Rooms", "Age"],
                "Value": [city, f"{net_sqm} m²", rooms, f"{age} Years"]
            }), hide_index=True, use_container_width=True)

    else:
        # Empty State
        st.info("👈 Enter property details and click 'Analyze' to generate a valuation.")
        st.markdown(
            """
            <div style="text-align: center; color: gray; margin-top: 50px;">
                <h1>🏡</h1>
                <p>Ready to predict</p>
            </div>
            """, unsafe_allow_html=True
        )

# --- 8. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Model: XGBoost Regressor")
    st.progress(100, text="Model Loaded")
    
    st.write("---")
    st.caption("Developed for Turkish Real Estate Market")
    st.caption("v1.0.2 | Local Mode")
