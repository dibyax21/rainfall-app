# Rainfall Predictor

Streamlit app that predicts daily rainfall (mm) from state, date, and historical normal rainfall.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Host on [Streamlit Community Cloud](https://share.streamlit.io): connect this repo and set main file to `app.py`.

## Retrain model

```bash
python train_model.py
```

Requires `rf_model_legacy.pkl` (original large model) if retraining from the teacher model.
