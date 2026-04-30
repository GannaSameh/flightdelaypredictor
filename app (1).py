import streamlit as st
import pandas as pd
import numpy as np
import requests, zipfile, io
from sklearn.model_selection import train_test_split, GridSearchCV
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

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark aviation theme */
[data-testid="stAppViewContainer"] {
    background: #0a0e1a;
    color: #e8eaf0;
}
[data-testid="stSidebar"] {
    background: #0f1628 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0f1628 0%, #1a2744 50%, #0d1f3c 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "✈";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 8rem;
    opacity: 0.06;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0 0 0.4rem 0;
    letter-spacing: -1px;
}
.hero p {
    color: #94a3b8;
    margin: 0;
    font-size: 1rem;
    font-weight: 300;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: #0f1628;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    text-align: center;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

/* Result banners */
.result-delayed {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 1px solid #dc2626;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.result-ontime {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 1px solid #16a34a;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.result-sub {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 0.4rem;
}

/* Sidebar labels */
.sidebar-section {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #3b82f6;
    text-transform: uppercase;
    margin: 1.5rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #1e3a5f;
}

/* Streamlit widget overrides */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── TRANSFORMER ───────────────────────────────────────────────────────────────
class SafeToString(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.copy().astype(str)

# ─── MODEL TRAINING ────────────────────────────────────────────────────────────
MODEL_PATH = "svm_model.pkl"
DATA_PATH  = "flights_clean.pkl"

@st.cache_resource(show_spinner=False)
def load_or_train():
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        model   = joblib.load(MODEL_PATH)
        flights = joblib.load(DATA_PATH)
        return model, flights

    with st.spinner("🛫  Downloading flight data & training SVM — please wait…"):
        url = "https://maven-datasets.s3.amazonaws.com/Airline+Flight+Delays/Airlines+Airports+Cancellation+Codes+%26+Flights.zip"
        r = requests.get(url, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        airlines    = pd.read_csv(z.open("airlines.csv"))
        flights_raw = pd.read_csv(z.open("flights.csv"))

    flights = flights_raw.sample(n=8000, random_state=42).reset_index(drop=True)
    flights = flights.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
    flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME"}, inplace=True)
    flights.drop(columns=["IATA_CODE"], inplace=True, errors="ignore")

    # Clean
    flights.drop_duplicates(inplace=True)
    flights = flights.loc[:, ~flights.columns.duplicated()]
    for col in flights.select_dtypes("object").columns:
        flights[col] = flights[col].fillna("UNKNOWN").astype(str).str.upper().str.strip()

    leakage = ["ARRIVAL_TIME","WHEELS_ON","TAXI_IN","ELAPSED_TIME","AIR_TIME",
               "DEPARTURE_TIME","WHEELS_OFF","AIR_SYSTEM_DELAY","SECURITY_DELAY",
               "AIRLINE_DELAY","LATE_AIRCRAFT_DELAY","WEATHER_DELAY","CANCELLATION_REASON"]
    flights.drop(columns=[c for c in leakage if c in flights.columns], inplace=True, errors="ignore")
    flights.drop(columns=["TAIL_NUMBER","FLIGHT_NUMBER"], inplace=True, errors="ignore")
    flights.loc[flights["CANCELLED"]==1, ["ARRIVAL_DELAY","DEPARTURE_DELAY"]] = 0
    for col in ["ARRIVAL_DELAY","DEPARTURE_DELAY"]:
        flights[col] = flights[col].clip(flights[col].quantile(0.01), flights[col].quantile(0.99))

    # Feature engineering
    flights["DELAYED"]     = (flights["ARRIVAL_DELAY"] > 15).astype(int)
    flights["HOUR"]        = flights["SCHEDULED_DEPARTURE"] // 100
    flights["IS_WEEKEND"]  = (flights["DAY_OF_WEEK"] >= 6).astype(int)
    flights["FLIGHT_TYPE"] = np.where(flights["DISTANCE"] < 1000, "SHORT", "LONG")

    X = flights.drop(["DELAYED","ARRIVAL_DELAY"], axis=1)
    y = flights["DELAYED"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=["int64","float64"]).columns
    cat_cols = X.select_dtypes(include="object").columns

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp",  SimpleImputer(strategy="most_frequent")),
            ("safe", SafeToString()),
            ("enc",  OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    pipe = Pipeline([("prep", preprocessor), ("clf", SVC(probability=True, C=10, gamma="scale", kernel="rbf"))])
    pipe.fit(X_train, y_train)

    # Metrics
    preds  = pipe.predict(X_test)
    probas = pipe.predict_proba(X_test)[:,1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1":       round(f1_score(y_test, preds), 4),
        "auc":      round(roc_auc_score(y_test, probas), 4),
    }
    joblib.dump((pipe, metrics, list(X.columns), X), MODEL_PATH)
    joblib.dump(flights, DATA_PATH)
    return (pipe, metrics, list(X.columns), X), flights

result, flights = load_or_train()
model, metrics, feature_cols, X_ref = result

# ─── SIDEBAR INPUTS ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-section">✈ Route Info</div>', unsafe_allow_html=True)

    airlines_list  = sorted(flights["AIRLINE_NAME"].dropna().unique().tolist())
    origins_list   = sorted(flights["ORIGIN_AIRPORT"].dropna().unique().tolist()) if "ORIGIN_AIRPORT" in flights else ["UNKNOWN"]
    dest_list      = sorted(flights["DESTINATION_AIRPORT"].dropna().unique().tolist()) if "DESTINATION_AIRPORT" in flights else ["UNKNOWN"]

    airline  = st.selectbox("Airline",      airlines_list)
    origin   = st.selectbox("Origin Airport",      origins_list[:200])
    dest     = st.selectbox("Destination Airport", dest_list[:200])

    st.markdown('<div class="sidebar-section">📅 Schedule</div>', unsafe_allow_html=True)
    month        = st.slider("Month",          1, 12, 6)
    day          = st.slider("Day of Month",   1, 31, 15)
    day_of_week  = st.slider("Day of Week",    1,  7,  3)
    sched_dep    = st.number_input("Scheduled Departure (HHMM)", min_value=0, max_value=2359, value=800, step=100)
    sched_arr    = st.number_input("Scheduled Arrival  (HHMM)", min_value=0, max_value=2359, value=1000, step=100)

    st.markdown('<div class="sidebar-section">🛫 Flight Details</div>', unsafe_allow_html=True)
    distance     = st.number_input("Distance (miles)", min_value=0, max_value=5000, value=800, step=50)
    dep_delay    = st.number_input("Departure Delay (min)", min_value=-60, max_value=300, value=0)
    taxi_out     = st.number_input("Taxi-Out Time (min)", min_value=0, max_value=120, value=15)
    cancelled    = st.selectbox("Cancelled?", [0, 1])
    diverted     = st.selectbox("Diverted?",  [0, 1])

    predict_btn  = st.button("🔍  PREDICT DELAY")

# ─── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>FLIGHT DELAY PREDICTOR</h1>
    <p>SVM-powered predictions on US domestic flight data &nbsp;|&nbsp; 15,000 flight training set</p>
</div>
""", unsafe_allow_html=True)

# ─── MODEL METRICS ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="val">{metrics["accuracy"]*100:.1f}%</div><div class="lbl">Accuracy</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="val">{metrics["f1"]*100:.1f}%</div><div class="lbl">F1 Score</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="val">{metrics["auc"]*100:.1f}%</div><div class="lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="val">SVM</div><div class="lbl">Model (RBF kernel)</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── PREDICTION ────────────────────────────────────────────────────────────────
if predict_btn:
    hour        = sched_dep // 100
    is_weekend  = int(day_of_week >= 6)
    flight_type = "SHORT" if distance < 1000 else "LONG"

    # Build a row matching training columns
    input_dict = {c: 0 for c in feature_cols}
    mapping = {
        "MONTH": month, "DAY": day, "DAY_OF_WEEK": day_of_week,
        "SCHEDULED_DEPARTURE": sched_dep, "SCHEDULED_ARRIVAL": sched_arr,
        "DEPARTURE_DELAY": dep_delay, "DISTANCE": distance,
        "TAXI_OUT": taxi_out, "CANCELLED": cancelled, "DIVERTED": diverted,
        "HOUR": hour, "IS_WEEKEND": is_weekend,
        "AIRLINE_NAME": airline, "FLIGHT_TYPE": flight_type,
    }
    if "ORIGIN_AIRPORT"      in input_dict: input_dict["ORIGIN_AIRPORT"]      = origin
    if "DESTINATION_AIRPORT" in input_dict: input_dict["DESTINATION_AIRPORT"] = dest
    if "AIRLINE_x"           in input_dict: input_dict["AIRLINE_x"]           = airline[:2]

    for k, v in mapping.items():
        if k in input_dict:
            input_dict[k] = v

    df_input = pd.DataFrame([input_dict])

    pred  = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0][1]

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if pred == 1:
            st.markdown(f"""
            <div class="result-delayed">
                <div style="font-size:3rem">⚠️</div>
                <div class="result-title" style="color:#fca5a5">DELAY PREDICTED</div>
                <div class="result-sub">This flight is likely to arrive more than 15 minutes late.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ontime">
                <div style="font-size:3rem">✅</div>
                <div class="result-title" style="color:#86efac">ON TIME</div>
                <div class="result-sub">This flight is expected to arrive within 15 minutes of schedule.</div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="metric-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
            <div class="val" style="font-size:2.5rem">{proba*100:.1f}%</div>
            <div class="lbl">Delay Probability</div>
            <div style="margin-top:1rem;background:#1e3a5f;border-radius:8px;height:10px;overflow:hidden;">
                <div style="background:{'#dc2626' if proba>0.5 else '#16a34a'};width:{proba*100:.0f}%;height:100%;border-radius:8px;transition:width 0.5s;"></div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Input summary
    with st.expander("📋  Input Summary"):
        summary = pd.DataFrame({
            "Feature": ["Airline","Origin","Destination","Month","Day of Week",
                        "Dep. Time","Arr. Time","Distance","Dep. Delay","Flight Type","Weekend"],
            "Value":   [airline, origin, dest, month, day_of_week,
                        sched_dep, sched_arr, f"{distance} mi", f"{dep_delay} min", flight_type,
                        "Yes" if is_weekend else "No"]
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

else:
    st.info("👈  Configure the flight parameters in the sidebar, then click **PREDICT DELAY**.")

    # Show a sample of the training data
    with st.expander("📊  Training Data Preview"):
        show_cols = ["AIRLINE_NAME","ORIGIN_AIRPORT","DESTINATION_AIRPORT",
                     "DISTANCE","DEPARTURE_DELAY","DELAYED"]
        show_cols = [c for c in show_cols if c in flights.columns]
        st.dataframe(flights[show_cols].head(20), use_container_width=True)
