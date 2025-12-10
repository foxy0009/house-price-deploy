import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. LOAD MODEL & SCALER ---
# This runs once and caches the result so it doesn't reload on every click
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Make sure .joblib and .pkl files are in the repo.")
        return None, None, None

model, scaler, model_columns = load_artifacts()

# --- 2. VIBE CONFIG ---
st.set_page_config(page_title="House Price AI", layout="centered")
st.title("🏡 House Price Predictor")
st.caption("Powered by XGBoost (Running Locally)")

# --- 3. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📍 Location & Details")
    city = st.selectbox("City", ["Istanbul", "Ankara", "Izmir", "Antalya", "Other"])
    heating = st.selectbox("Heating Type", ["Kombi Doğalgaz", "Merkezi Doğalgaz", "Other"])
    
    st.divider()
    st.write("🔧 Technical Specs")
    age = st.slider("Building Age", 0, 50, 5)
    floor = st.number_input("Floor Number", min_value=-3, value=2)
    total_floors = st.number_input("Total Floors in Building", min_value=1, value=5)

# --- 4. MAIN PAGE INPUTS ---
col1, col2 = st.columns(2)
with col1:
    net_sqm = st.number_input("Net Area (m²)", min_value=10, value=100)
    rooms = st.number_input("Rooms", min_value=1.0, value=3.0, step=0.5)

with col2:
    bathrooms = st.number_input("Bathrooms", min_value=0, value=1)

# --- 5. PREDICTION LOGIC ---
def predict_price():
    if model is None:
        return
    
    # Create input dictionary
    input_data = {
        "Net_Metrekare": net_sqm,
        "Oda_Sayısı": rooms,
        "Bulunduğu_Kat": floor,
        "Binanın_Yaşı": age,
        "Binanın_Kat_Sayısı": total_floors,
        # Banyo_Sayısı is ignored by model but we kept the input for UI
    }
    
    # 1. Base DataFrame
    df = pd.DataFrame([input_data])
    
    # 2. One-Hot Encoding (Manual)
    sehir_col = f"Şehir_{city.lower()}" if city != "Other" else "other"
    heat_col = f"Isıtma_Tipi_{heating}" if heating != "Other" else "other"
    
    # 3. Match Columns
    df = df.reindex(columns=model_columns, fill_value=0)
    
    if sehir_col in df.columns: df[sehir_col] = 1
    if heat_col in df.columns: df[heat_col] = 1
    
    # 4. Scale
    scale_cols = ['Net_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 
                  'Binanın_Yaşı', 'Binanın_Kat_Sayısı']
    df[scale_cols] = scaler.transform(df[scale_cols])
    
    # 5. Predict
    log_pred = model.predict(df)[0]
    price = np.expm1(log_pred)
    
    return price

# --- 6. BUTTON ---
if st.button("✨ Predict Price", type="primary", use_container_width=True):
    predicted_price = predict_price()
    if predicted_price:
        st.balloons()
        st.success(f"💰 Estimated Price: {predicted_price:,.0f} TL")