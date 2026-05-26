"""Vercel entrypoint — FastAPI ASGI app (Streamlit does not run on Vercel)."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model_utils import MONTHLY_NORMAL_HINT, STATES, predict_rainfall

app = FastAPI(title="Rainfall Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    state_code: int = Field(ge=1, le=36)
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)
    normal: float = Field(ge=0.0, le=500.0)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/states")
def states():
    return {
        "states": [{"code": c, "name": n} for c, n in sorted(STATES.items(), key=lambda x: x[1])],
        "monthly_hints": MONTHLY_NORMAL_HINT,
    }


@app.post("/api/predict")
def predict(body: PredictRequest):
    try:
        return predict_rainfall(
            body.state_code, body.year, body.month, body.day, body.normal
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found") from None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
