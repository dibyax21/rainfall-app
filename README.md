# Rainfall Predictor

Streamlit app that predicts daily rainfall (mm) from state, date, and historical normal rainfall.

## Run locally (Streamlit)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Vercel

`app.py` exports a FastAPI `app` for Vercel’s Python runtime. The UI is in `public/index.html`.

1. Import the repo at [vercel.com](https://vercel.com)
2. Redeploy after each push to `main`

API: `POST /api/predict` · UI: `/`

## Deploy on Streamlit Cloud

1. Open [https://share.streamlit.io](https://share.streamlit.io) and click **New app**.
2. Select repository: **`dibyax21/rainfall-app`**
3. Branch: **`main`**
4. Main file path: **`streamlit_app.py`** (not `app.py`)
5. Click **Deploy**.

Community Cloud will auto-redeploy when you push new commits to `main`.

## Retrain model

```bash
python train_model.py
```

Requires `rf_model_legacy.pkl` (original large model) if retraining from the teacher model.
