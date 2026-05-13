"""
Rain Prediction Dashboard — Streamlit App
==========================================
Requirements:
    pip install streamlit tensorflow keras joblib scikit-learn numpy pandas plotly

File structure expected in the same directory as this script:
    app.py
    tcn_rain_model.keras
    scaler.pkl
    feature_cols.pkl
    recent_history.csv

To run:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime
import os
import plotly.graph_objects as go

# Model Files Directory
MODEL_DIR = "models"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rain Predictor",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Background ── */
.stApp {
    background: linear-gradient(160deg, #0d1b2a 0%, #1b2d45 45%, #0f2233 100%);
    background-attachment: fixed;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
section[data-testid="stSidebar"] { display: none; }

/* ── Constrain page width & centre ── */
.main .block-container {
    max-width: 1100px;
    padding: 2rem 2.5rem 3rem;
    margin: 0 auto;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #e8f4fd;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.1;
}
.hero h1 em { color: #5bc8f5; }
.hero p {
    color: #8ab4cc;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* ── Date badge ── */
.date-badge {
    display: inline-block;
    background: rgba(91,200,245,0.10);
    border: 1px solid rgba(91,200,245,0.22);
    border-radius: 999px;
    padding: 0.28rem 1rem;
    font-size: 0.8rem;
    color: #5bc8f5;
    letter-spacing: 0.4px;
    margin: 0.8rem 0 2rem;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #5bc8f5;
    margin-bottom: 0.9rem;
}

/* ── Parameter panel ── */
.param-panel {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.8rem 2rem 1.4rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
}

/* ── Input labels ── */
label { color: #9fc8e0 !important; font-size: 0.82rem !important; }

/* ── Number input boxes ── */
input[type="number"] {
    background: rgba(13,27,42,0.7) !important;
    border: 1px solid rgba(91,200,245,0.18) !important;
    border-radius: 10px !important;
    color: #e8f4fd !important;
    font-size: 0.92rem !important;
}
input[type="number"]:focus {
    border-color: rgba(91,200,245,0.5) !important;
    box-shadow: 0 0 0 2px rgba(91,200,245,0.12) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1565a8 0%, #2a90d4 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem 2rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 24px rgba(42,144,212,0.32) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(42,144,212,0.48) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Stat pills ── */
.stat-row {
    display: flex;
    gap: 0.8rem;
    margin-top: 0.6rem;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
}
.stat-pill {
    flex: 1;
    min-width: 100px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    text-align: center;
}
.stat-pill .val { font-size: 1.4rem; color: #5bc8f5; font-weight: 600; }
.stat-pill .lbl { font-size: 0.68rem; color: #8ab4cc; margin-top: 0.15rem; letter-spacing: 0.5px; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(91,200,245,0.2), transparent);
    margin: 2rem 0;
}

/* ── Result card ── */
.result-card {
    border-radius: 20px;
    padding: 2.2rem 2rem;
    text-align: center;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.result-rain {
    background: linear-gradient(135deg, rgba(20,70,120,0.65), rgba(20,110,170,0.45));
    border: 1px solid rgba(91,200,245,0.35);
    box-shadow: 0 0 50px rgba(91,200,245,0.10);
}
.result-norain {
    background: linear-gradient(135deg, rgba(30,80,50,0.65), rgba(50,130,70,0.45));
    border: 1px solid rgba(80,200,120,0.35);
    box-shadow: 0 0 50px rgba(80,200,120,0.10);
}
.result-placeholder {
    background: rgba(255,255,255,0.025);
    border: 1px dashed rgba(255,255,255,0.1);
}
.result-icon   { font-size: 3.8rem; margin-bottom: 0.4rem; }
.result-label  { font-family:'DM Serif Display',serif; font-size:1.9rem; color:#e8f4fd; margin:0.2rem 0; }
.result-prob   { font-size:0.95rem; color:#a0c8e0; font-weight:300; }

/* ── Interpretation bar ── */
.interp-bar {
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    font-size: 0.87rem;
    margin-top: 0.6rem;
    border-left-width: 3px;
    border-left-style: solid;
    background: rgba(255,255,255,0.04);
}

/* ── Metrics ── */
[data-testid="stMetricValue"] { color: #e8f4fd !important; }
[data-testid="stMetricLabel"] { color: #8ab4cc !important; font-size:0.8rem !important; }

/* ── Note box ── */
.note-box {
    background: rgba(91,200,245,0.06);
    border-left: 3px solid rgba(91,200,245,0.35);
    border-radius: 0 10px 10px 0;
    padding: 0.6rem 0.9rem;
    font-size: 0.78rem;
    color: #8ab4cc;
    margin-top: 0.8rem;
}

/* ── Alert ── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── Artifact loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    files = ["tcn_rain_model.keras", "scaler.pkl",
             "feature_cols.pkl", "recent_history.csv"]
    missing = [f for f in files if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        return None, None, None, None, missing
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    def last_timestep(t):
        return t[:, -1, :]

    model = load_model(
        os.path.join(MODEL_DIR, "tcn_rain_model.h5"),
        custom_objects={"last_timestep": last_timestep},
        safe_mode=False,
    )
    scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    history_df   = pd.read_csv(os.path.join(MODEL_DIR, "recent_history.csv"), index_col=0)
    return model, scaler, feature_cols, history_df, []

model, scaler, feature_cols, history_df, missing_files = load_artifacts()

AUTO_COLS = {"sin_doy", "cos_doy", "gap", "observed"}


# ── Gauge ─────────────────────────────────────────────────────────────────────
def make_gauge(prob: float, is_rain: bool) -> go.Figure:
    color = "#2a90d4" if is_rain else "#3dc47e"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 34, "color": "#e8f4fd", "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1,
                     "tickcolor": "rgba(255,255,255,0.18)",
                     "tickfont": {"color": "#8ab4cc", "size": 10}},
            "bar": {"color": color, "thickness": 0.26},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "rgba(61,196,126,0.10)"},
                {"range": [30, 60], "color": "rgba(245,200,65,0.08)"},
                {"range": [60,100], "color": "rgba(42,144,212,0.13)"},
            ],
            "threshold": {
                "line": {"color": "rgba(255,255,255,0.35)", "width": 2},
                "thickness": 0.75, "value": 50,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=16, b=8, l=16, r=16), height=200,
    )
    return fig


# ── Feature range hints ───────────────────────────────────────────────────────
FEATURE_HINTS: dict[str, dict] = {
    "T0700":        {"min":-20.0, "max":55.0,   "default":25.0,  "step":0.1,  "unit":"°C", "label": "Suhu pada pukul 07:00"},
    "T1300":        {"min":-20.0, "max":55.0,   "default":25.0,  "step":0.1,  "unit":"°C", "label": "Suhu pada pukul 13:00"},
    "T1800":        {"min":-20.0, "max":55.0,   "default":25.0,  "step":0.1,  "unit":"°C", "label": "Suhu pada pukul 18:00"},
    "Tmax":         {"min":-20.0, "max":55.0,   "default":30.0,  "step":0.1,  "unit":"°C", "label": "Suhu Maksimum"},
    "Tmin":         {"min":-20.0, "max":55.0,   "default":20.0,  "step":0.1,  "unit":"°C", "label": "Suhu Minimum"},
    "Trata-rata":   {"min":-20.0, "max":55.0,   "default":25.0,  "step":0.1,  "unit":"°C", "label": "Suhu Rata-Rata"},
    "RH0700":       {"min":0.0,   "max":100.0,  "default":70.0,  "step":0.5,  "unit":"%", "label": "Kelembaban pada pukul 07:00"},
    "RH1300":       {"min":0.0,   "max":100.0,  "default":70.0,  "step":0.5,  "unit":"%", "label": "Kelembaban pada pukul 13:00"},
    "RH1800":       {"min":0.0,   "max":100.0,  "default":70.0,  "step":0.5,  "unit":"%", "label": "Kelembaban pada pukul 18:00"},
    "RHrata-rata":  {"min":0.0,   "max":100.0,  "default":70.0,  "step":0.5,  "unit":"%", "label": "Kelembaban Rata-Rata"},
    "QFF (tekanan udara)":             {"min":950.0, "max":1050.0, "default":1013.0,"step":0.1,  "unit":"hPa", "label": "Tekanan Udara"},
    "ffrata-rata (kecepatan angin)":   {"min":0.0,   "max":100.0,  "default":10.0,  "step":0.1,  "unit":"km/jam", "label": "Kecepatan Angin Rata-Rata"},
    "dd (arah angin)":                 {"min":0.0,   "max":360.0,  "default":180.0, "step":1.0,  "unit":"°", "label": "Arah Angin"},
    "LPM":                             {"min":0.0,   "max":16.0,   "default":6.0,   "step":0.1,  "unit":"jam", "label": "Lama Penyinaran Matahari"},
}

def get_hint(col: str) -> dict:
    key = col.strip()
    for k, v in FEATURE_HINTS.items():
        if k in key or key in k:
            return v
    return {"min": -1000.0, "max": 1000.0, "default": 0.0, "step": 0.01, "unit": ""}


# ═════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Dashboard Prediksi <em>Hujan</em></h1>
    <p>Temporal Convolutional Network · Meteorological Binary Classifier</p>
</div>
""", unsafe_allow_html=True)

# today_str = datetime.date.today().strftime("%A, %B %d %Y")
# st.markdown(
#     f'<div style="text-align:center">'
#     f'<span class="date-badge">📅 {today_str}</span>'
#     f'</div>',
#     unsafe_allow_html=True,
# )

# ── Missing files guard ───────────────────────────────────────────────────────
if missing_files:
    st.error(
        f"**Missing model files:** `{'`, `'.join(missing_files)}`\n\n"
        "Place these files in the same directory as `app.py`.\n\n"
        "**Export from your Colab notebook:**\n"
        "```python\n"
        "import joblib\n"
        "model.save('tcn_rain_model.keras')\n"
        "joblib.dump(scaler, 'scaler.pkl')\n"
        "joblib.dump(scale_cols, 'feature_cols.pkl')\n"
        "recent = df_tcn[df_tcn['observed']==1].tail(44)[scale_cols]\n"
        "recent.to_csv('recent_history.csv')\n"
        "```"
    )
    st.stop()

# ── Collect user columns ──────────────────────────────────────────────────────
user_cols = [c for c in feature_cols if c not in AUTO_COLS]

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — Parameter inputs (2-column grid)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Parameter Input Cuaca Hari Ini</div>',
            unsafe_allow_html=True)
# st.markdown('<div class="param-panel">', unsafe_allow_html=True)

user_values: dict = {}
N_GRID = 2
rows = [user_cols[i : i + N_GRID] for i in range(0, len(user_cols), N_GRID)]

for row_group in rows:
    grid_cols = st.columns(N_GRID, gap="medium")
    for col_widget, col_name in zip(grid_cols, row_group):
        hint  = get_hint(col_name)
        label = hint.get("label", col_name.replace("_", " ").title())
        if hint["unit"]:
            label = f"{label} ({hint['unit']})"
        with col_widget:
            user_values[col_name] = st.number_input(
                label,
                min_value=float(hint["min"]),
                max_value=float(hint["max"]),
                value=float(hint["default"]),
                step=float(hint["step"]),
                key=f"input_{col_name}",
                format="%g",
            )

st.markdown('</div>', unsafe_allow_html=True)  # /param-panel

lookback = (len(history_df) + 1) if history_df is not None else 45

st.markdown(f"""
    <div class="note-box">
        💡 Input Anda secara otomatis digabungkan dengan data historis yang tersimpan selama <strong>{lookback - 1} hari terakhir</strong>
        untuk membentuk interval {lookback} hari sebelum prediksi dilakukan.
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict button — centred, prominent ──────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button("Prediksi Hujan Hari Ini", use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Results
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Hasil Prediksi</div>', unsafe_allow_html=True)

res_left, res_right = st.columns([1, 1.1], gap="large")

# Left: stat pills + input summary table
with res_left:
    st.markdown(f"""
        <div class="stat-row">
            <div class="stat-pill">
                <div class="val">{lookback}</div>
                <div class="lbl">Hari dalam Interval</div>
            </div>
            <div class="stat-pill">
                <div class="val">{len(user_cols)}</div>
                <div class="lbl">Parameter Input</div>
            </div>
            <div class="stat-pill">
                <div class="val">TCN</div>
                <div class="lbl">Tipe Model</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label" style="margin-top:0.2rem;">Ringkasan Input</div>',
                unsafe_allow_html=True)
    preview_df = pd.DataFrame({
        "Parameter": [c.replace("_", " ").title() for c in user_values],
        "Nilai":     [round(v, 3) for v in user_values.values()],
    }).set_index("Parameter")
    st.dataframe(
        preview_df.style
            .format("{:g}")
            .set_properties(**{
                "background-color": "rgba(255,255,255,0.02)",
                "color": "#c8e0f0",
                "border": "1px solid rgba(255,255,255,0.05)",
            }),
        use_container_width=True,
        height=min(38 * len(user_values) + 38, 440),
    )

# Right: prediction output
with res_right:
    if not predict_btn:
        st.markdown("""
            <div class="result-card result-placeholder" style="padding:3.5rem 2rem;">
                <div style="font-size:3.5rem; opacity:0.3; margin-bottom:1rem;">🌦️</div>
                <div style="font-size:0.95rem; color:#8ab4cc; font-weight:300; line-height:1.8;">
                    Masukkan nilai cuaca hari ini pada kotak input di atas<br>dan tekan tombol
                    <strong style="color:#5bc8f5;">Prediksi Hujan Hari Ini</strong>.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Running model…"):
            try:
                doy = datetime.date.today().timetuple().tm_yday
                today_row = dict(user_values)
                today_row["sin_doy"]  = float(np.sin(2 * np.pi * doy / 365.25))
                today_row["cos_doy"]  = float(np.cos(2 * np.pi * doy / 365.25))
                today_row["observed"] = 1.0
                today_row["gap"]      = 0.0

                today_df  = pd.DataFrame([today_row])[feature_cols]
                window_df = pd.concat(
                    [history_df[feature_cols], today_df], ignore_index=True
                ).tail(45)

                window_scaled = scaler.transform(window_df.to_numpy(np.float32))
                prob     = float(model.predict(window_scaled[np.newaxis, ...], verbose=0)[0][0])
                is_rain  = prob >= 0.5
                conf     = prob if is_rain else (1 - prob)

                rc   = "result-rain" if is_rain else "result-norain"
                icon = "🌧️" if is_rain else "☀️"
                lbl  = "Diprediksi Hujan" if is_rain else "Diprediksi Tidak Hujan"

                st.markdown(f"""
                    <div class="result-card {rc}">
                        <div class="result-icon">{icon}</div>
                        <div class="result-label">{lbl}</div>
                        <div class="result-prob">{conf:.1%} model confidence</div>
                    </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(make_gauge(prob, is_rain), use_container_width=True,
                                config={"displayModeBar": False})

                if prob < 0.3:
                    interp, ic = "☀️ Probabilitas hujan rendah.", "#3dc47e"
                elif prob < 0.5:
                    interp, ic = "⛅ Probabilitas ringan.", "#a8d86e"
                elif prob < 0.7:
                    interp, ic = "🌦️ Probabilitas sedang.", "#f5c842"
                else:
                    interp, ic = "🌧️ Probabilitas hujan tinggi.", "#5bc8f5"

                st.markdown(f"""
                    <div class="interp-bar" style="border-left-color:{ic}; color:{ic};">
                        {interp}
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("Probabilitas Hujan",    f"{prob:.1%}")
                m2.metric("Probabilitas Tidak Hujan", f"{1 - prob:.1%}")

            except Exception as e:
                st.error(f"**Prediction failed:** {e}")
                st.info("Check that `recent_history.csv` columns match `feature_cols.pkl`.")


# ── Footer ────────────────────────────────────────────────────────────────────
# st.markdown("<br>", unsafe_allow_html=True)
# st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
# st.markdown("""
#     <div style='text-align:center; color:#2e4a60; font-size:0.73rem; padding-bottom:0.5rem;'>
#         TCN Rain Predictor · Temporal Convolutional Network · Meteorological Binary Classifier
#     </div>
# """, unsafe_allow_html=True)