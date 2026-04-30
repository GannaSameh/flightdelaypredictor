import streamlit as st
import pandas as pd
import numpy as np
import requests, zipfile, io, gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib, os, warnings

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flight Predictor", page_icon="✈️")

# ─── TRANSFORMER ───────────────────────────────────────────────────────────────
class SafeToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.copy().astype(str)

# ─── MICRO-DATA LOADER ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train():
    MODEL_PATH = "svm_model.pkl"
    DATA_PATH  = "flights_clean.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        return joblib.load(MODEL_PATH), joblib.load(DATA_PATH)

    with st.status("🛸 Launching Micro-Engine...", expanded=True) as status:
        url = "https://maven-datasets.s3.amazonaws.com/Airline+Flight+Delays/Airlines+Airports+Cancellation+Codes+%26+Flights.zip"
        
        # Stream and extract only what we need
        r = requests.get(url, timeout=120)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # Load airlines first (small)
        airlines = pd.read_csv(z.open("airlines.csv"))
        
        # Load flights and immediately downsample to 1,000 rows
        status.write("📉 Applying Micro-Sampling (1,000 rows)...")
        flights_raw = pd.read_csv(z.open("flights.csv"))
        flights = flights_raw.sample(n=1000, random_state=42).reset_index(drop=True)
        
        # FORCED CLEANUP: Remove the large raw dataframe from RAM immediately
        del flights_raw
        gc.collect() 
        
        # Merge and Pre-process
        flights = flights.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
        flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"}, inplace=True)
        
        # Minimal cleaning to keep overhead low
        leakage = ["ARRIVAL_TIME","DEPARTURE_TIME","AIR_SYSTEM_DELAY","SECURITY_DELAY","AIRLINE_DELAY"]
        flights.drop(columns=[c for c in leakage if c in flights.columns], inplace=True, errors="ignore")
        
        flights["DELAYED"] = (flights["ARRIVAL_DELAY"] > 15).astype(int)
        flights["HOUR"] = (flights["SCHEDULED_DEPARTURE"] // 100).fillna(0)
        
        X = flights[["AIRLINE_NAME", "DISTANCE", "HOUR", "DAY_OF_WEEK", "MONTH"]]
        y = flights["DELAYED"]
        
        status.write("🧠 Training SVM...")
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), ["DISTANCE", "HOUR", "DAY_OF_WEEK", "MONTH"]),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("safe", SafeToString()), ("enc", OneHotEncoder(handle_unknown="ignore"))]), ["AIRLINE_NAME"])
        ])

        pipe = Pipeline([("prep", preprocessor), ("clf", SVC(probability=True, C=0.5))])
        pipe.fit(X, y)

        metrics = {"accuracy": round(accuracy_score(y, pipe.predict(X)), 2)}
        result = (pipe, metrics, list(X.columns), X)
        
        joblib.dump(result, MODEL_PATH)
        joblib.dump(flights, DATA_PATH)
        
        status.update(label="✅ Ready!", state="complete")
        return result, flights

# ─── RUN APP ──────────────────────────────────────────────────────────────────
try:
    result, flights = load_or_train()
    model, metrics, feature_cols, X_ref = result
    st.success(f"App Loaded! Model Accuracy: {metrics['accuracy']*100}%")
except Exception as e:
    st.error(f"App failed to start: {e}")

# [Simplified Sidebar/Predict Logic]
