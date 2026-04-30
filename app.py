import streamlit as st
import pandas as pd
import numpy as np
import requests, zipfile, io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import joblib, os, warnings

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# ─── HELPERS & TRANSFORMERS ────────────────────────────────────────────────────
class SafeToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.copy().astype(str)

def download_and_extract():
    """Handles the heavy lifting of downloading large datasets."""
    url = "https://maven-datasets.s3.amazonaws.com/Airline+Flight+Delays/Airlines+Airports+Cancellation+Codes+%26+Flights.zip"
    try:
        # 120s timeout to prevent Cloud health-check resets during download
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        return pd.read_csv(z.open("airlines.csv")), pd.read_csv(z.open("flights.csv"))
    except Exception as e:
        st.error(f"Data retrieval failed: {e}")
        st.stop()

# ─── DATA & MODEL LOADING ──────────────────────────────────────────────────────
MODEL_PATH = "svm_model.pkl"
DATA_PATH  = "flights_clean.pkl"

@st.cache_resource(show_spinner=False)
def load_or_train():
    # If the app crashed previously, pre-saved files prevent re-triggering the crash
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        try:
            return joblib.load(MODEL_PATH), joblib.load(DATA_PATH)
        except:
            pass 

    with st.status("🚀 Initializing Aviation Intelligence...", expanded=True) as status:
        status.write("📥 Downloading Maven Dataset...")
        airlines, flights_raw = download_and_extract()

        status.write("🧹 Cleaning and Downsampling (Memory Optimization)...")
        # Reducing sample size to 5000 ensures SVM fits in Streamlit Cloud's 1GB RAM
        flights = flights_raw.sample(n=5000, random_state=42).reset_index(drop=True)
        flights = flights.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
        
        # Cleanup to free memory
        del flights_raw 
        
        flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"}, inplace=True)
        for col in flights.select_dtypes("object").columns:
            flights[col] = flights[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()

        # Leakage prevention
        leakage = ["ARRIVAL_TIME","WHEELS_ON","TAXI_IN","ELAPSED_TIME","AIR_TIME",
                   "DEPARTURE_TIME","WHEELS_OFF","AIR_SYSTEM_DELAY","SECURITY_DELAY",
                   "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","CANCELLATION_REASON"]
        flights.drop(columns=[c for c in leakage if c in flights.columns], inplace=True, errors="ignore")
        
        # Feature Engineering
        flights["DELAYED"] = (flights["ARRIVAL_DELAY"] > 15).astype(int)
        flights["HOUR"] = flights["SCHEDULED_DEPARTURE"] // 100
        flights["IS_WEEKEND"] = (flights["DAY_OF_WEEK"] >= 6).astype(int)
        flights["FLIGHT_TYPE"] = np.where(flights["DISTANCE"] < 1000, "SHORT", "LONG")

        X = flights.drop(["DELAYED","ARRIVAL_DELAY"], axis=1)
        y = flights["DELAYED"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        status.write("🧠 Training SVM (RBF Kernel)...")
        num_cols = X.select_dtypes(include=["int64","float64"]).columns
        cat_cols = X.select_dtypes(include="object").columns

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("safe", SafeToString()), ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ])

        # C=1 is much faster and less memory-intensive than C=10
        pipe = Pipeline([("prep", preprocessor), ("clf", SVC(probability=True, C=1, cache_size=500))])
        pipe.fit(X_train, y_train)

        metrics = {
            "accuracy": round(accuracy_score(y_test, pipe.predict(X_test)), 4),
            "f1":       round(f1_score(y_test, pipe.predict(X_test)), 4),
            "auc":      round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]), 4),
        }
        
        result = (pipe, metrics, list(X.columns), X)
        joblib.dump(result, MODEL_PATH)
        joblib.dump(flights, DATA_PATH)
        
        status.update(label="✈️ System Ready!", state="complete", expanded=False)
        return result, flights

# Execute
result, flights = load_or_train()
model, metrics, feature_cols, X_ref = result

# Rest of your UI code...
st.title("Flight Delay Predictor")
st.write(f"Model Accuracy: {metrics['accuracy']}")
