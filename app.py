import streamlit as st
import pandas as pd
import numpy as np
import requests, zipfile, io, gc
import joblib, os, warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Flight Delay Predictor", page_icon="✈️", layout="wide")

# ─── TRANSFORMER (Verbatim from Colab) ────────────────────────────────────────
class SafeToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        return X.astype(str)

# ─── MEMORY-EFFICIENT ENGINE ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train_fast():
    MODEL_PATH = "best_svm_cloud.pkl"
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    with st.status("🛸 Initializing Cloud-Optimized Engine...", expanded=True) as status:
        # Data Retrieval
        url = "https://maven-datasets.s3.amazonaws.com/Airline+Flight+Delays/Airlines+Airports+Cancellation+Codes+%26+Flights.zip"
        r = requests.get(url, timeout=120)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        airlines = pd.read_csv(z.open("airlines.csv"))
        flights_raw = pd.read_csv(z.open("flights.csv"))

        # MICRO-SAMPLE: Reduced from 15,000 to 2,500 for Cloud stability
        flights = flights_raw.sample(n=2500, random_state=42).reset_index(drop=True)
        del flights_raw
        gc.collect()

        # Merging & Cleaning[cite: 2]
        flights = flights.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
        flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"}, inplace=True)
        
        for col in flights.select_dtypes(include="object").columns:
            flights[col] = flights[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()

        # Leakage & Feature Engineering[cite: 2]
        leakage = ["ARRIVAL_TIME","WHEELS_ON","TAXI_IN","ELAPSED_TIME","AIR_TIME",
                   "DEPARTURE_TIME","WHEELS_OFF","AIR_SYSTEM_DELAY","SECURITY_DELAY"]
        flights.drop(columns=[c for c in leakage if c in flights.columns], inplace=True, errors="ignore")
        
        flights["DELAYED"] = (flights["ARRIVAL_DELAY"] > 15).astype(int)
        flights["HOUR"] = flights["SCHEDULED_DEPARTURE"] // 100
        flights["IS_WEEKEND"] = (flights["DAY_OF_WEEK"] >= 6).astype(int)
        
        # Select key features to keep OHE matrix small
        features = ["AIRLINE_NAME", "DISTANCE", "HOUR", "DAY_OF_WEEK", "MONTH", "DEPARTURE_DELAY"]
        X = flights[features]
        y = flights["DELAYED"]

        # Pipeline Builder[cite: 2]
        num_cols = ["DISTANCE", "HOUR", "DAY_OF_WEEK", "MONTH", "DEPARTURE_DELAY"]
        cat_cols = ["AIRLINE_NAME"]

        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("safe", SafeToString()), ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
        ])

        # SINGLE FIT: No GridSearchCV to avoid OOM crash
        model = Pipeline([("prep", preprocessor), ("clf", SVC(probability=True, C=1, kernel='rbf'))])
        model.fit(X, y)
        
        joblib.dump((model, features), MODEL_PATH)
        status.update(label="✅ Cloud Engine Ready!", state="complete")
        return model, features

# ─── RUN APP ──────────────────────────────────────────────────────────────────
model, feature_cols = load_and_train_fast()

st.title("✈️ Smart Flight Delay Predictor")
st.info("Predictor running on a memory-optimized SVM (2,500 flight records).")

# Dynamic Inputs
with st.expander("📝 Enter Flight Details", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox("Airline", ["UNITED AIR LINES INC.", "AMERICAN AIRLINES INC.", "DELTA AIR LINES INC.", "SOUTHWEST AIRLINES CO."])
        dist = st.number_input("Distance (miles)", value=500)
        dep_delay = st.number_input("Departure Delay (min)", value=0)
    with col2:
        month = st.slider("Month", 1, 12, 6)
        dow = st.slider("Day of Week", 1, 7, 3)
        hour = st.slider("Hour of Day", 0, 23, 12)

if st.button("🔍 Predict Delay"):
    input_df = pd.DataFrame([[airline, dist, hour, dow, month, dep_delay]], columns=feature_cols)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    if pred == 1:
        st.error(f"Prediction: DELAYED ({prob:.1%} probability)")
    else:
        st.success(f"Prediction: ON TIME ({1-prob:.1%} probability)")
