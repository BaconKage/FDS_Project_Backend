# app.py — Flask backend for FDS Project
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os

# Plotly for charts
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)
CORS(app)

# ======================
# Data loading
# ======================
CSV_PATH = os.path.join(os.path.dirname(__file__), "health_activity_data_cleaned.csv")
if not os.path.exists(CSV_PATH):
    raise Exception("⚠ Put health_activity_data_cleaned.csv inside project/backend/")

df = pd.read_csv(CSV_PATH)
df = df.copy()
df.fillna(df.mean(numeric_only=True), inplace=True)

# ensure numeric types where relevant
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# ======================
# Helpers
# ======================
def to_float(series):
    return pd.to_numeric(series, errors="coerce")

def bmi_calc(weight_kg, height_cm):
    try:
        h = float(height_cm) / 100.0
        w = float(weight_kg)
        if h <= 0:
            return 0.0
        return round(w / (h * h), 2)
    except Exception:
        return 0.0

def style_plot(fig, title=None, height=420):
    if title:
        fig.update_layout(title=title, title_font=dict(size=18, color="#111827"))
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=30, t=60, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Inter, system-ui, -apple-system", size=13, color="#111827"),
        xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb"),
        yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb"),
        hovermode="closest",
    )
    return fig

# ======================
# Health check
# ======================
@app.get("/")
def health():
    return {"ok": True}

# ======================
# Predict endpoint (simple rule-based label to keep UI working)
# ======================
@app.post("/predict")
def predict():
    """
    Body:
    {
      "Age": 20, "Gender": "Male",
      "Height": 170, "Weight": 70,
      "SystolicBP": 120, "DiastolicBP": 80,
      "HeartRate": 72, "SleepDuration": 7
    }
    """
    d = request.get_json(force=True) or {}

    height = float(d.get("Height") or 0)
    weight = float(d.get("Weight") or 0)
    systolic = float(d.get("SystolicBP") or 0)
    diastolic = float(d.get("DiastolicBP") or 0)
    heart = float(d.get("HeartRate") or 0)
    sleep = float(d.get("SleepDuration") or 0)

    bmi = bmi_calc(weight, height)

    # simple rule-based label so UI always gets a result
    risk = (bmi >= 30) or (systolic >= 140) or (diastolic >= 90) or (heart >= 100) or (sleep < 5)
    label = "At Risk" if risk else "Healthy"

    return jsonify({"prediction": label, "bmi": bmi})

# ======================
# Dashboard endpoint (global KPIs + charts)
# ======================
@app.get("/dashboard")
def dashboard():
    count = int(len(df))

    avg_hr = to_float(df.get("Heart_Rate")).mean() if "Heart_Rate" in df.columns else None
    avg_sys = to_float(df.get("Blood_Pressure_Systolic")).mean() if "Blood_Pressure_Systolic" in df.columns else None
    avg_dia = to_float(df.get("Blood_Pressure_Diastolic")).mean() if "Blood_Pressure_Diastolic" in df.columns else None
    avg_sleep = to_float(df.get("Hours_of_Sleep")).mean() if "Hours_of_Sleep" in df.columns else None

    # dataset BMI
    df_bmi = df.copy()
    avg_bmi = None
    if {"Weight_kg", "Height_cm"}.issubset(df.columns):
        def _bmi_row(row):
            try:
                h = float(row["Height_cm"]) / 100.0
                w = float(row["Weight_kg"])
                if h <= 0:
                    return None
                return w / (h * h)
            except Exception:
                return None
        df_bmi["BMI"] = df_bmi.apply(_bmi_row, axis=1)
        avg_bmi = pd.to_numeric(df_bmi["BMI"], errors="coerce").mean()

    # ----- charts -> HTML strings -----
    # BMI histogram
    if "BMI" in df_bmi.columns:
        fig_bmi = px.histogram(df_bmi, x="BMI", nbins=30)
        style_plot(fig_bmi, "BMI Distribution")
        bmi_html = pio.to_html(fig_bmi, full_html=False)
    else:
        bmi_html = "<div>No BMI column to chart.</div>"

    # Heart Rate vs Sleep
    if {"Heart_Rate", "Hours_of_Sleep"}.issubset(df.columns):
        fig_hr_sleep = px.scatter(
            df, x="Hours_of_Sleep", y="Heart_Rate", opacity=0.6,
            labels={"Hours_of_Sleep":"Sleep (hrs)", "Heart_Rate":"Heart Rate (bpm)"}
        )
        style_plot(fig_hr_sleep, "Heart Rate vs Sleep Duration")
        hr_sleep_html = pio.to_html(fig_hr_sleep, full_html=False)
    else:
        hr_sleep_html = "<div>Missing Heart_Rate/Hours_of_Sleep for chart.</div>"

    # NEW: Systolic vs Diastolic scatter
    if {"Blood_Pressure_Systolic","Blood_Pressure_Diastolic"}.issubset(df.columns):
        fig_bp_scatter = px.scatter(
            df,
            x="Blood_Pressure_Systolic",
            y="Blood_Pressure_Diastolic",
            opacity=0.6,
            labels={"Blood_Pressure_Systolic":"Systolic (mmHg)","Blood_Pressure_Diastolic":"Diastolic (mmHg)"},
        )
        style_plot(fig_bp_scatter, "Systolic vs Diastolic")
        bp_scatter_html = pio.to_html(fig_bp_scatter, full_html=False)
    else:
        bp_scatter_html = "<div>Missing BP columns for chart.</div>"

    # NEW: Heart Rate histogram
    if "Heart_Rate" in df.columns:
        fig_hr_hist = px.histogram(df, x="Heart_Rate", nbins=30, labels={"Heart_Rate":"Heart Rate (bpm)"})
        style_plot(fig_hr_hist, "Heart Rate Distribution")
        hr_hist_html = pio.to_html(fig_hr_hist, full_html=False)
    else:
        hr_hist_html = "<div>Missing Heart_Rate for chart.</div>"

    # NEW: Sleep histogram
    if "Hours_of_Sleep" in df.columns:
        fig_sleep_hist = px.histogram(df, x="Hours_of_Sleep", nbins=24, labels={"Hours_of_Sleep":"Sleep (hrs)"})
        style_plot(fig_sleep_hist, "Sleep Duration Distribution")
        sleep_hist_html = pio.to_html(fig_sleep_hist, full_html=False)
    else:
        sleep_hist_html = "<div>Missing Hours_of_Sleep for chart.</div>"

    cards = [
        {"title": "Records", "value": count, "unit": ""},
        {"title": "Avg BMI", "value": None if avg_bmi is None else round(float(avg_bmi), 2), "unit": ""},
        {"title": "Avg Systolic", "value": None if avg_sys is None else round(float(avg_sys), 1), "unit": "mmHg"},
        {"title": "Avg Diastolic", "value": None if avg_dia is None else round(float(avg_dia), 1), "unit": "mmHg"},
        {"title": "Avg Heart Rate", "value": None if avg_hr is None else round(float(avg_hr), 1), "unit": "bpm"},
        {"title": "Avg Sleep", "value": None if avg_sleep is None else round(float(avg_sleep), 1), "unit": "hrs"},
    ]

    return jsonify({
        "cards": cards,
        "charts": {
            "bmi": bmi_html,
            "hr_sleep": hr_sleep_html,
            "bp_scatter": bp_scatter_html,
            "hr_hist": hr_hist_html,
            "sleep_hist": sleep_hist_html,
        }
    })

# ======================
# Compare endpoint (user vs dataset + personalized charts)
# ======================
@app.post("/compare")
def compare():
    """
    Body: { Height, Weight, SystolicBP, DiastolicBP, HeartRate, SleepDuration }
    Returns per-metric comparisons + Plotly charts with user's marker.
    """
    d = request.get_json(force=True) or {}

    def num(key):
        try:
            return float(d.get(key, 0))
        except Exception:
            return 0.0

    user = {
        "Height_cm": num("Height"),
        "Weight_kg": num("Weight"),
        "Blood_Pressure_Systolic": num("SystolicBP"),
        "Blood_Pressure_Diastolic": num("DiastolicBP"),
        "Heart_Rate": num("HeartRate"),
        "Hours_of_Sleep": num("SleepDuration"),
    }
    user_bmi = bmi_calc(user["Weight_kg"], user["Height_cm"])

    df_num = df.copy()
    for c in df_num.columns:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    def percentile(col, val):
        s = pd.to_numeric(df_num.get(col), errors="coerce").dropna()
        if len(s) == 0:
            return None
        pct = (s <= val).mean() * 100.0
        return round(float(pct), 1)

    comparisons = {}
    comp_cols = [
        ("Height_cm", "Height (cm)"),
        ("Weight_kg", "Weight (kg)"),
        ("Blood_Pressure_Systolic", "Systolic BP"),
        ("Blood_Pressure_Diastolic", "Diastolic BP"),
        ("Heart_Rate", "Heart Rate"),
        ("Hours_of_Sleep", "Sleep (hrs)"),
    ]

    for col, label in comp_cols:
        if col in df_num.columns:
            s = to_float(df_num[col])
            avg = float(s.mean())
            u = float(user[col])
            diff = u - avg
            comparisons[col] = {
                "label": label,
                "user_value": round(u, 2),
                "dataset_avg": round(avg, 2),
                "difference": round(diff, 2),
                "percentile": percentile(col, u),
                "status": "Above average" if diff > 0 else "Below average" if diff < 0 else "On average",
            }

    # BMI comparison (computed)
    if {"Weight_kg", "Height_cm"}.issubset(df_num.columns):
        s_bmi = df_num.apply(lambda r: bmi_calc(r.get("Weight_kg"), r.get("Height_cm")), axis=1)
        s_bmi = pd.to_numeric(s_bmi, errors="coerce").dropna()
        if len(s_bmi):
            avg_bmi = float(s_bmi.mean())
            comparisons["BMI"] = {
                "label": "BMI",
                "user_value": user_bmi,
                "dataset_avg": round(avg_bmi, 2),
                "difference": None if user_bmi is None else round(user_bmi - avg_bmi, 2),
                "percentile": None if user_bmi is None else round((s_bmi <= user_bmi).mean() * 100.0, 1),
                "status": None if user_bmi is None else ("Above average" if user_bmi > avg_bmi else "Below average" if user_bmi < avg_bmi else "On average"),
            }

    # Charts with user's marker
    charts = {}

    # BMI histogram with vline
    try:
        df_bmi = df_num.copy()
        df_bmi["BMI"] = df_bmi.apply(lambda r: bmi_calc(r.get("Weight_kg"), r.get("Height_cm")), axis=1)
        fig_bmi = px.histogram(df_bmi, x="BMI", nbins=30)
        if user_bmi is not None:
            fig_bmi.add_vline(
                x=user_bmi, line_dash="dash", line_color="red",
                annotation_text=f"Your BMI: {user_bmi}", annotation_position="top left"
            )
        style_plot(fig_bmi, "BMI Distribution (You vs Dataset)")
        charts["bmi"] = pio.to_html(fig_bmi, full_html=False)
    except Exception as e:
        charts["bmi"] = f"<div>Could not render BMI chart: {e}</div>"

    # HR vs Sleep scatter with user's point
    try:
        if {"Hours_of_Sleep", "Heart_Rate"}.issubset(df_num.columns):
            fig2 = px.scatter(
                df_num, x="Hours_of_Sleep", y="Heart_Rate", opacity=0.6,
                labels={"Hours_of_Sleep": "Sleep (hrs)", "Heart_Rate": "Heart Rate (bpm)"}
            )
            fig2.add_scatter(
                x=[user["Hours_of_Sleep"]], y=[user["Heart_Rate"]],
                mode="markers+text", text=["You"], textposition="top center",
                marker=dict(color="red", size=12, symbol="star"), name="You"
            )
            style_plot(fig2, "Heart Rate vs Sleep (You vs Dataset)")
            charts["hr_sleep"] = pio.to_html(fig2, full_html=False)
        else:
            charts["hr_sleep"] = "<div>Missing columns for chart.</div>"
    except Exception as e:
        charts["hr_sleep"] = f"<div>Could not render HR/Sleep chart: {e}</div>"

    # NEW: Systolic vs Diastolic with your point
    try:
        if {"Blood_Pressure_Systolic","Blood_Pressure_Diastolic"}.issubset(df_num.columns):
            fig_bp2 = px.scatter(
                df_num,
                x="Blood_Pressure_Systolic", y="Blood_Pressure_Diastolic",
                opacity=0.6,
                labels={"Blood_Pressure_Systolic":"Systolic (mmHg)","Blood_Pressure_Diastolic":"Diastolic (mmHg)"},
            )
            fig_bp2.add_scatter(
                x=[user["Blood_Pressure_Systolic"]],
                y=[user["Blood_Pressure_Diastolic"]],
                mode="markers+text",
                text=["You"], textposition="top center",
                marker=dict(color="red", size=12, symbol="star"), name="You"
            )
            style_plot(fig_bp2, "Systolic vs Diastolic (You vs Dataset)")
            charts["bp_scatter"] = pio.to_html(fig_bp2, full_html=False)
        else:
            charts["bp_scatter"] = "<div>Missing BP columns for chart.</div>"
    except Exception as e:
        charts["bp_scatter"] = f"<div>Could not render BP scatter: {e}</div>"

    # NEW: HR histogram with your vline
    try:
        if "Heart_Rate" in df_num.columns:
            fig_hrh = px.histogram(df_num, x="Heart_Rate", nbins=30, labels={"Heart_Rate":"Heart Rate (bpm)"})
            fig_hrh.add_vline(x=user["Heart_Rate"], line_dash="dash", line_color="red",
                              annotation_text=f"Your HR: {user['Heart_Rate']}", annotation_position="top left")
            style_plot(fig_hrh, "Heart Rate Distribution (You vs Dataset)")
            charts["hr_hist"] = pio.to_html(fig_hrh, full_html=False)
        else:
            charts["hr_hist"] = "<div>Missing Heart_Rate.</div>"
    except Exception as e:
        charts["hr_hist"] = f"<div>Could not render HR hist: {e}</div>"

    # NEW: Sleep histogram with your vline
    try:
        if "Hours_of_Sleep" in df_num.columns:
            fig_sh = px.histogram(df_num, x="Hours_of_Sleep", nbins=24, labels={"Hours_of_Sleep":"Sleep (hrs)"})
            fig_sh.add_vline(x=user["Hours_of_Sleep"], line_dash="dash", line_color="red",
                             annotation_text=f"Your Sleep: {user['Hours_of_Sleep']}", annotation_position="top left")
            style_plot(fig_sh, "Sleep Duration Distribution (You vs Dataset)")
            charts["sleep_hist"] = pio.to_html(fig_sh, full_html=False)
        else:
            charts["sleep_hist"] = "<div>Missing Hours_of_Sleep.</div>"
    except Exception as e:
        charts["sleep_hist"] = f"<div>Could not render Sleep hist: {e}</div>"

    return jsonify({"comparisons": comparisons, "charts": charts})

# ======================
# Run
# ======================
if __name__ == "__main__":
    # 127.0.0.1 to match your frontend fetches
    app.run(host="127.0.0.1", port=5000, debug=True)
