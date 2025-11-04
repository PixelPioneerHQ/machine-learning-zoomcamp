from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import os
import pickle


class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float


app = FastAPI(title="Lead Conversion Service")

# Resolve model path robustly across local/uv/Docker runs.
# Priority:
# 1) MODEL_PATH env var
# 2) pipeline_v2.bin next to this file (Docker base image)
# 3) pipeline_v1.bin next to this file
# 4) pipeline_v1.bin in the parent directory (e.g., ../pipeline_v1.bin)
# 5) pipeline_v1.bin in current working directory
env_path = os.getenv("MODEL_PATH")
base_dir = Path(__file__).resolve().parent
candidates = [
    Path(env_path) if env_path else None,
    base_dir / "pipeline_v2.bin",
    base_dir / "pipeline_v1.bin",
    base_dir.parent / "pipeline_v1.bin",
    Path.cwd() / "pipeline_v1.bin",
]

model_path = None
for p in candidates:
    if p and p.exists():
        model_path = p
        break

if model_path is None:
    tried = [str(p) for p in candidates if p is not None]
    raise FileNotFoundError("Model file not found. Tried: " + ", ".join(tried))

with open(model_path, "rb") as f:
    model = pickle.load(f)


@app.get("/")
async def root():
    return {"status": "ok", "model_file": str(model_path)}


@app.post("/predict")
async def predict(payload: Lead):
    proba = float(model.predict_proba([payload.model_dump()])[:, 1][0])
    return {"converted_probability": proba}

