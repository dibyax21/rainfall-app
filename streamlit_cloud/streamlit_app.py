import pandas as pd
import streamlit as st

from model_utils import MONTHLY_NORMAL_HINT, MONTH_NAMES, STATES, predict_rainfall


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
        st.markdown("1. Pick your state\n2. Choose a date\n3. Set normal rainfall\n4. Tap **Get prediction**")
        st.caption("Model: Random Forest")

    st.markdown(
        '<div class="hero"><h1>🌧️ Rainfall Predictor</h1>'
        "<p>Simple, clear forecasts for any state and date in India.</p></div>",
        unsafe_allow_html=True,
    )

    state_names = sorted(STATES.values())
    default_index = state_names.index(STATES[15])

    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### 📍 Location & date")
        with st.container(border=True):
            selected_name = st.selectbox("State / region", state_names, index=default_index)
            state_code = next(c for c, n in STATES.items() if n == selected_name)
            picked = st.date_input("Date", value=pd.Timestamp("2026-05-22"))
            year, month, day = picked.year, picked.month, picked.day
            st.caption(f"Selected: **{MONTH_NAMES[month]} {day}, {year}** · {selected_name}")

    with right:
        st.markdown("#### 💧 Climate input")
        with st.container(border=True):
            hint = MONTHLY_NORMAL_HINT[month]
            st.markdown(f"Suggested for {MONTH_NAMES[month]}: **~{hint:.0f} mm**")
            normal_rainfall = st.slider("Normal rainfall (mm)", 0.0, 150.0, float(hint), 0.5)

    if st.button("Get rainfall prediction", type="primary", use_container_width=True):
        result = predict_rainfall(state_code, year, month, day, normal_rainfall)
        cat = result["category"]
        st.success("🎉 Prediction completed!")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted rainfall", f"{result['prediction']:.1f} mm")
        c2.metric("Normal (baseline)", f"{result['normal']:.1f} mm")
        c3.metric("Vs normal", f"{result['delta']:+.1f} mm")
        st.markdown(
            f'<div class="result-box" style="background:linear-gradient(135deg,{cat["color"]},#1e3a8a);">'
            f'<div style="font-size:2.5rem;">{cat["emoji"]}</div>'
            f'<div class="result-value">{result["prediction"]:.1f} mm</div>'
            f'<div class="result-label">{cat["label"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("👆 Fill in the form, then click **Get rainfall prediction**.")


if __name__ == "__main__":
    main()

