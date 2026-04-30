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
    layout="wide",
    initial_sidebar_state="expanded"
)

# [CSS Block remains the same as your original code]
st.markdown("""
<style>
/* ... (Your CSS styles here) ... */
</style>
""", unsafe_allow_html=True)

# ─── TRANSFORMER ───────────────────────────────────────────────────────────────
class SafeToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.copy().astype(str)

# ─── DOWNLOAD HELPER ──────────────────────────────────────────────────────────
def download_data():
    """Robust download with progress status and timeout handling."""
    url = "https://maven-datasets.s3.amazonaws.com/Airline+Flight+Delays/Airlines+Airports+Cancellation+Codes+%26+Flights.zip"
    
    with st.status("📡 Connecting to data server...", expanded=True) as status:
        try:
            # Using stream=True to handle large files better
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            
            status.update(label="📦 Extracting flight records...", state="running")
            z = zipfile.ZipFile(io.BytesIO(response.content))
            
            airlines = pd.read_csv(z.open("airlines.csv"))
            flights_raw = pd.read_csv(z.open("flights.csv"))
            
            status.update(label="✅ Data downloaded successfully!", state="complete", expanded=False)
            return airlines, flights_raw
        
        except requests.exceptions.Timeout:
            status.update(label="❌ Download timed out. Please refresh the page.", state="error")
            st.error("The data server took too long to respond. Streamlit Cloud may have limited bandwidth.")
            st.stop()
        except Exception as e:
            status.update(label=f"❌ Error: {str(e)}", state="error")
            st.stop()

# ─── MODEL TRAINING ────────────────────────────────────────────────────────────
MODEL_PATH = "svm_model.pkl"
DATA_PATH  = "flights_clean.pkl"

@st.cache_resource(show_spinner=False)
def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        try:
            model_data = joblib.load(MODEL_PATH)
            flights = joblib.load(DATA_PATH)
            return model_data, flights
        except:
            pass # If files are corrupt, re-train

    # Call the robust downloader
    airlines, flights_raw = download_data()

    with st.spinner("⚙️ Processing data and training SVM..."):
        # Downsample further if Streamlit Cloud crashes (memory limit)
        flights = flights_raw.sample(n=7000, random_state=42).reset_index(drop=True)
        flights = flights.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
        
        # ... [Data Cleaning logic remains same as your original] ...
        flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"}, inplace=True)
        flights.drop(columns=["IATA_CODE"], inplace=True, errors="ignore")
        flights.drop_duplicates(inplace=True)
        for col in flights.select_dtypes("object").columns:
            flights[col] = flights[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()

        leakage = ["ARRIVAL_TIME","WHEELS_ON","TAXI_IN","ELAPSED_TIME","AIR_TIME",
                   "DEPARTURE_TIME","WHEELS_OFF","AIR_SYSTEM_DELAY","SECURITY_DELAY",
                   "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","CANCELLATION_REASON"]
        flights.drop(columns=[c for c in leakage if c in flights.columns], inplace=True, errors="ignore")
        flights.drop(columns=["TAIL_NUMBER","FLIGHT_NUMBER"], inplace=True, errors="ignore")
        
        flights["DELAYED"]     = (flights["ARRIVAL_DELAY"] > 15).astype(int)
        flights["HOUR"]        = flights["SCHEDULED_DEPARTURE"] // 100
        flights["IS_WEEKEND"]  = (flights["DAY_OF_WEEK"] >= 6).astype(int)
        flights["FLIGHT_TYPE"] = np.where(flights["DISTANCE"] < 1000, "SHORT", "LONG")

        X = flights.drop(["DELAYED","ARRIVAL_DELAY"], axis=1)
        y = flights["DELAYED"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        num_cols = X.select_dtypes(include=["int64","float64"]).columns
        cat_cols = X.select_dtypes(include="object").columns

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("safe", SafeToString()), ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ])

        pipe = Pipeline([("prep", preprocessor), ("clf", SVC(probability=True, C=1, kernel="rbf"))]) # Reduced C for faster training
        pipe.fit(X_train, y_train)

        metrics = {
            "accuracy": round(accuracy_score(y_test, pipe.predict(X_test)), 4),
            "f1":       round(f1_score(y_test, pipe.predict(X_test)), 4),
            "auc":      round(roc_auc_score(y_test, pipe.predict_proba(X_test)[:,1]), 4),
        }
        
        model_output = (pipe, metrics, list(X.columns), X)
        joblib.dump(model_output, MODEL_PATH)
        joblib.dump(flights, DATA_PATH)
        
        return model_output, flights

# Run Loader
result, flights = load_or_train()
model, metrics, feature_cols, X_ref = result

# ... [Sidebar and Prediction logic remains the same] ...
