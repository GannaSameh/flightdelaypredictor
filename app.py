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

# [CSS remains the same as app (1).py]
st.markdown("""<style>/* Your CSS here */</style>""", unsafe_allow_html=True)

# ─── TRANSFORMER ───────────────────────────────────────────────────────────────
class SafeToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.copy().astype(str)

# ─── ROBUST DATA LOADER ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train():
    MODEL_PATH = "svm_model.pkl"
    DATA_PATH  = "flights_clean.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        try:
            return joblib.load(MODEL_PATH), joblib.load(DATA_PATH)
        except: pass

    with st.status("🛠️ Initializing Light-Weight Model...", expanded=True) as status:
        status.write("📡 Fetching flight records...")
        url = "https://maven-datasets.s3.amazonaws.com/Airline+Flight+Delays/Airlines+Airports+Cancellation+Codes+%26+Flights.zip"
        
        try:
            r = requests.get(url, timeout=120)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            airlines = pd.read_csv(z.open("airlines.csv"))
            # Stream directly into a smaller sample to save RAM
            flights_raw = pd.read_csv(z.open("flights.csv"))
            status.write("✂️ Reducing dataset for stability...")
            
            # SMALL SAMPLE: 2,000 rows is very safe for Streamlit Cloud
            flights = flights_raw.sample(n=2000, random_state=42).reset_index(drop=True)
            
            # FREE MEMORY IMMEDIATELY
            del flights_raw
            gc.collect() 
            
            flights = flights.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
            flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"}, inplace=True)
            
            status.write("🧹 Cleaning data...")
            for col in flights.select_dtypes("object").columns:
                flights[col] = flights[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()

            leakage = ["ARRIVAL_TIME","WHEELS_ON","TAXI_IN","ELAPSED_TIME","AIR_TIME",
                       "DEPARTURE_TIME","WHEELS_OFF","AIR_SYSTEM_DELAY","SECURITY_DELAY",
                       "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","CANCELLATION_REASON"]
            flights.drop(columns=[c for c in leakage if c in flights.columns], inplace=True, errors="ignore")
            
            flights["DELAYED"] = (flights["ARRIVAL_DELAY"] > 15).astype(int)
            flights["HOUR"] = flights["SCHEDULED_DEPARTURE"] // 100
            flights["IS_WEEKEND"] = (flights["DAY_OF_WEEK"] >= 6).astype(int)
            flights["FLIGHT_TYPE"] = np.where(flights["DISTANCE"] < 1000, "SHORT", "LONG")

            X = flights.drop(["DELAYED","ARRIVAL_DELAY"], axis=1)
            y = flights["DELAYED"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            status.write("🧠 Training SVM (Optimized)...")
            num_cols = X.select_dtypes(include=["int64","float64"]).columns
            cat_cols = X.select_dtypes(include="object").columns

            preprocessor = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("safe", SafeToString()), ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
            ])

            # C=1 and limited sample makes this training very fast
            pipe = Pipeline([("prep", preprocessor), ("clf", SVC(probability=True, C=1))])
            pipe.fit(X_train, y_train)

            metrics = {
                "accuracy": round(accuracy_score(y_test, pipe.predict(X_test)), 4),
                "f1":       round(f1_score(y_test, pipe.predict(X_test)), 4),
                "auc":      round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]), 4),
            }

            result = (pipe, metrics, list(X.columns), X)
            joblib.dump(result, MODEL_PATH)
            joblib.dump(flights, DATA_PATH)
            
            status.update(label="✅ Setup Complete!", state="complete", expanded=False)
            return result, flights
            
        except Exception as e:
            st.error(f"Critical Error: {e}")
            st.stop()

# ─── MAIN EXECUTION ────────────────────────────────────────────────────────────
result, flights = load_or_train()
model, metrics, feature_cols, X_ref = result

# [Sidebar and Hero section from app (1).py]
st.title("Flight Delay Predictor")
st.write(f"Model Accuracy (Small Sample): {metrics['accuracy']*100:.1f}%")

# [Rest of prediction logic...]
