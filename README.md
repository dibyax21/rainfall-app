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

Connect this repo at [share.streamlit.io](https://share.streamlit.io) and set main file to **`streamlit_app.py`** (not `app.py`).

## Retrain model

```bash
python train_model.py
```

Requires `rf_model_legacy.pkl` (original large model) if retraining from the teacher model.
