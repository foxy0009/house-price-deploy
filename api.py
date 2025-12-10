import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(title="House Price Predictor API")

# --- 1. Load Artifacts ---
try:
    model = joblib.load('xgb_model.joblib')
    scaler = joblib.load('scaler.pkl')
    # Use the columns list saved during training to ensure exact order
    model_columns = joblib.load('model_columns.pkl')
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    model = None


# --- 2. Input Schema ---
# We keep Banyo_Sayısı here so the website (Streamlit) works,
# but we will ignore it in the logic below.
class HouseInput(BaseModel):
    Net_Metrekare: float = Field(..., gt=10, description="Net area in m2")
    Oda_Sayısı: float = Field(..., gt=0, description="Number of rooms")
    Banyo_Sayısı: float = Field(..., ge=0, description="Number of bathrooms")  # Kept for compatibility
    Binanın_Yaşı: float = Field(..., ge=0, description="Age of building (0-8 encoded)")
    Binanın_Kat_Sayısı: int = Field(..., gt=0, description="Total floors in building")
    Bulunduğu_Kat: int = Field(..., description="Floor of the flat")

    # String inputs
    Sehir: str = Field(..., example="antalya", description="City name")
    Isitma_Tipi: str = Field(..., example="Kombi Doğalgaz", description="Heating type")


# --- 3. Preprocessing ---
def _preprocess_input(input_data: HouseInput) -> pd.DataFrame:
    data_dict = input_data.dict()

    # --- THE FIX: REMOVE BATHROOMS ---
    # We remove 'Banyo_Sayısı' from the data dictionary immediately.
    # The model will never see what the user typed.
    if "Banyo_Sayısı" in data_dict:
        del data_dict["Banyo_Sayısı"]

    # 1. Normalize Inputs
    # Lowercase city to match training keys (e.g. "Şehir_antalya")
    sehir_input = data_dict.pop("Sehir").lower()
    # Heating type usually keeps casing (e.g. "Isıtma_Tipi_Kombi Doğalgaz")
    isitma_input = data_dict.pop("Isitma_Tipi")

    # 2. Create Base DataFrame
    df = pd.DataFrame([data_dict])

<<<<<<< Updated upstream
    # 2. Derived numerical features used in training
    df["Avg_Room_Size"] = df["Net_Metrekare"] / np.maximum(df["Oda_Sayısı"], 0.5)

    # 3. Defaults for categorical groups that are not user-facing
    defaults = {
        "Tapu_Durumu_Unknown": 1,
        "Kullanım_Durumu_Mülk Sahibi Oturuyor": 1,
        "Kullanım_Durumu_Kiracı Oturuyor": 0,
        "Takas_Yok": 1,
        "Yatırıma_Uygunluk_Unknown": 1,
        "Eşya_Durumu_Unknown": 1,
    }
    for col, value in defaults.items():
        df[col] = value

    # 4. One-hot encode city/heating according to training columns
    # 2. One-hot encode city/heating according to training columns
    city_cols = [c for c in model_columns if c.startswith("Şehir_")]
    heat_cols = [c for c in model_columns if c.startswith("Isıtma_Tipi_")]

    for col in city_cols + heat_cols:
        df[col] = 0

    target_city_col = f"Şehir_{data_dict['Sehir']}"
    target_heat_col = f"Isıtma_Tipi_{data_dict['Isitma_Tipi']}"

    if target_city_col in city_cols:
        df[target_city_col] = 1
    if target_heat_col in heat_cols:
        df[target_heat_col] = 1

    # 5. Ensure all model columns exist before scaling/ordering
    df = df.reindex(columns=list(set(model_columns) | set(df.columns)), fill_value=0)

    # 6. Scaling using the fitted scaler feature set
    features_to_scale = list(getattr(scaler, "feature_names_in_", []))
    if not features_to_scale:
        features_to_scale = [
            'Net_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat',
            'Binanın_Yaşı', 'Binanın_Kat_Sayısı', 'Banyo_Sayısı', 'Avg_Room_Size'
        ]

    missing_scale_cols = [col for col in features_to_scale if col not in df.columns]
    for col in missing_scale_cols:
        df[col] = 0

    df[features_to_scale] = scaler.transform(df[features_to_scale])

    # 7. Final alignment to model column order
    df = df.reindex(columns=model_columns, fill_value=0)
    # 3. Align with Model Columns (creates any other missing columns and orders correctly)
=======
    # 3. Handle One-Hot Encoding (Manual Creation)
    target_city_col = f"Şehir_{sehir_input}"
    target_heat_col = f"Isıtma_Tipi_{isitma_input}"

    # 4. Reindex to match model structure exactly
    # This ensures we have all columns the model expects, filled with 0s initially
>>>>>>> Stashed changes
    df = df.reindex(columns=model_columns, fill_value=0)

    # 5. Set the active One-Hot columns to 1
    if target_city_col in df.columns:
        df[target_city_col] = 1

    if target_heat_col in df.columns:
        df[target_heat_col] = 1

    # 6. Scaling
    # We REMOVED 'Banyo_Sayısı' from this list so the scaler doesn't crash
    scale_cols = [
        'Net_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat',
        'Binanın_Yaşı', 'Binanın_Kat_Sayısı'
    ]

    # Filter to ensure we only scale columns that are actually in the dataframe
    cols_to_scale = [c for c in scale_cols if c in df.columns]

    if cols_to_scale:
        try:
            df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        except ValueError as e:
            # If the scaler still expects bathrooms (because you didn't retrain yet),
            # this might warn, but it usually tries to scale what it can.
            print(f"Scaling Warning: {e}")

    return df


# --- 4. Endpoint ---
@app.post("/predict")
def predict(house: HouseInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        processed_df = _preprocess_input(house)

        # XGBoost prediction
        log_pred = model.predict(processed_df)[0]
        real_price = np.expm1(log_pred)  # Reverse log transformation

        return {
            "prediction_tl": round(float(real_price), 2),
            "status": "success"
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)