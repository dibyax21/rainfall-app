import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "rf_model.pkl"

STATES = {
    1: "Andhra Pradesh",
    2: "Arunachal Pradesh",
    3: "Assam",
    4: "Bihar",
    5: "Chhattisgarh",
    6: "Goa",
    7: "Gujarat",
    8: "Haryana",
    9: "Himachal Pradesh",
    10: "Jammu & Kashmir",
    11: "Jharkhand",
    12: "Karnataka",
    13: "Kerala",
    14: "Madhya Pradesh",
    15: "Maharashtra",
    16: "Manipur",
    17: "Meghalaya",
    18: "Mizoram",
    19: "Nagaland",
    20: "Odisha",
    21: "Punjab",
    22: "Rajasthan",
    23: "Sikkim",
    24: "Tamil Nadu",
    25: "Telangana",
    26: "Tripura",
    27: "Uttar Pradesh",
    28: "Uttarakhand",
    29: "West Bengal",
    30: "Andaman & Nicobar",
    31: "Chandigarh",
    32: "Delhi",
    33: "Ladakh",
    34: "Lakshadweep",
    35: "Puducherry",
    36: "Dadra & Nagar Haveli",
}

MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Typical daily normal rainfall (mm) by month — rough India-wide guide for the slider default
MONTHLY_NORMAL_HINT = {
    1: 3.0, 2: 4.0, 3: 6.0, 4: 12.0, 5: 18.0, 6: 35.0,
    7: 45.0, 8: 40.0, 9: 28.0, 10: 15.0, 11: 8.0, 12: 4.0,
}


def rain_category(mm: float) -> tuple[str, str, str]:
    if mm < 2.5:
        return "Trace / dry", "🌤️", "#94A3B8"
    if mm < 15:
        return "Light rain", "🌦️", "#38BDF8"
    if mm < 64:
        return "Moderate rain", "🌧️", "#2563EB"
    if mm < 124:
        return "Heavy rain", "⛈️", "#1E40AF"
    return "Very heavy rain", "🌩️", "#1E3A8A"


@st.cache_resource(show_spinner="Loading prediction model…")
def load_model():
    return joblib.load(MODEL_PATH)


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        .block-container { padding-top: 1.5rem; max-width: 920px; }
        .hero {
            background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 55%, #1e3a8a 100%);
            border-radius: 20px;
            padding: 2rem 2rem 1.75rem;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 12px 40px rgba(37, 99, 235, 0.25);
        }
        .hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.35rem 0; color: white; }
        .hero p { margin: 0; opacity: 0.92; font-size: 1.05rem; }
        .card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 0.75rem;
        }
        .result-box {
            border-radius: 18px;
            padding: 1.5rem;
            text-align: center;
            color: white;
            margin-top: 0.5rem;
        }
        .result-value { font-size: 2.75rem; font-weight: 700; line-height: 1.1; }
        .result-label { font-size: 1.1rem; opacity: 0.9; margin-top: 0.35rem; }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f0f9ff 0%, #f8fafc 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Rainfall Predictor",
        page_icon="🌧️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    with st.sidebar:
        st.markdown("### About this app")
        st.markdown(
            "Predict **expected rainfall (mm)** for a location using "
            "state, date, and the historical **normal** rainfall for that day."
        )
        st.markdown("---")
        st.markdown("**How to use**")
        st.markdown("1. Pick your state\n2. Choose a date\n3. Set normal rainfall (or use the suggested value)\n4. Tap **Get prediction**")
        st.markdown("---")
        st.caption("Model: Random Forest · Trained on regional climate patterns")

    st.markdown(
        """
        <div class="hero">
            <h1>🌧️ Rainfall Predictor</h1>
            <p>Simple, clear forecasts for any state and date in India.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        model = load_model()
    except FileNotFoundError:
        st.error("Model file `rf_model.pkl` is missing. Run `python train_model.py` first.")
        st.stop()

    state_names = sorted(STATES.values())
    default_state = STATES[15]  # Maharashtra — common demo default
    default_index = state_names.index(default_state)

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### 📍 Location & date")
        with st.container(border=True):
            selected_name = st.selectbox(
                "State / region",
                state_names,
                index=default_index,
                help="Choose the state you want a forecast for.",
            )
            state_code = next(code for code, name in STATES.items() if name == selected_name)

            picked = st.date_input(
                "Date",
                value=pd.Timestamp("2026-05-22"),
                min_value=pd.Timestamp("2000-01-01"),
                max_value=pd.Timestamp("2100-12-31"),
                help="Pick the day you want to predict rainfall for.",
            )
            year, month, day = picked.year, picked.month, picked.day

            st.caption(f"Selected: **{MONTH_NAMES[month]} {day}, {year}** · {selected_name}")

    with right:
        st.markdown("#### 💧 Climate input")
        with st.container(border=True):
            hint = MONTHLY_NORMAL_HINT[month]
            st.markdown(
                f"**Normal rainfall** is the long-term average (mm) for this time of year. "
                f"Suggested for {MONTH_NAMES[month]}: **~{hint:.0f} mm**"
            )
            normal_rainfall = st.slider(
                "Normal rainfall (mm)",
                min_value=0.0,
                max_value=150.0,
                value=float(hint),
                step=0.5,
                help="Adjust if you know the local IMD or historical average for this day.",
            )

    st.markdown("")

    predict = st.button("Get rainfall prediction", type="primary", use_container_width=True)

    if predict:
        input_data = pd.DataFrame(
            [{
                "state_code": state_code,
                "year": year,
                "month": month,
                "day": day,
                "normal": normal_rainfall,
            }]
        )
        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, prediction)
        delta = prediction - normal_rainfall
        label, emoji, color = rain_category(prediction)

        st.markdown("---")
        st.markdown("#### Your forecast")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted rainfall", f"{prediction:.1f} mm")
        c2.metric("Normal (baseline)", f"{normal_rainfall:.1f} mm")
        c3.metric("Vs normal", f"{delta:+.1f} mm", delta_color="normal" if delta >= 0 else "inverse")

        st.markdown(
            f"""
            <div class="result-box" style="background: linear-gradient(135deg, {color} 0%, #1e3a8a 100%);">
                <div style="font-size: 2.5rem;">{emoji}</div>
                <div class="result-value">{prediction:.1f} mm</div>
                <div class="result-label">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("What does this mean?", expanded=False):
            st.markdown(
                f"- **{label}** — based on standard daily rainfall bands.\n"
                f"- Your prediction is **{abs(delta):.1f} mm {'above' if delta >= 0 else 'below'}** "
                f"the normal you entered.\n"
                f"- Inputs used: {selected_name}, {MONTH_NAMES[month]} {day}, {year}, normal {normal_rainfall:.1f} mm."
            )

        chart_df = pd.DataFrame({
            "Type": ["Normal (baseline)", "Predicted"],
            "Rainfall (mm)": [normal_rainfall, prediction],
        })
        st.bar_chart(chart_df.set_index("Type"), color="#2563EB", height=280)

    else:
        st.info("👆 Fill in the form above, then click **Get rainfall prediction** to see your forecast.")


if __name__ == "__main__":
    main()
