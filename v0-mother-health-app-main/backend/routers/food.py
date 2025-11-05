import io
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.applications.efficientnet import preprocess_input

from backend.data.labels_food101 import LABELS

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "food101_effnetb0_224.keras"
NUTRI_PATH = ROOT / "data" / "nutrition_map.json"
IMG_SIZE = 224
MAX_CLASSES = len(LABELS)

if not MODEL_PATH.exists():
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
with open(NUTRI_PATH, "r", encoding="utf-8") as f:
    NUTRI = json.load(f)


class FoodTopK(BaseModel):
    label: str
    score: float


class FoodResponse(BaseModel):
    best_label: str
    topk: List[FoodTopK]
    nutrition_per_portion: Optional[dict] = None


def prep_bytes(raw: bytes) -> np.ndarray:
    pil = Image.open(io.BytesIO(raw)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(pil, dtype=np.float32)
    x = preprocess_input(x)
    x = x.astype(np.float16)
    return np.expand_dims(x, 0)


def build_nutrition(best_label: str, portion_g: int) -> Optional[dict]:
    info = NUTRI.get(best_label)
    if not info or "per_100g" not in info:
        return None

    factor = portion_g / 100.0
    per100 = info["per_100g"]
    return {
        "portion_g": portion_g,
        "calories": round(per100.get("calories", 0.0) * factor, 1),
        "carbs_g": round(per100.get("carbs_g", 0.0) * factor, 1),
        "fat_g": round(per100.get("fat_g", 0.0) * factor, 1),
        "protein_g": round(per100.get("protein_g", 0.0) * factor, 1),
    }


@router.post("/food", response_model=FoodResponse)
async def classify_food(
    image: UploadFile = File(...),
    portion_g: int = Form(200),
    k: int = Form(5),
) -> FoodResponse:
    if portion_g <= 0:
        raise HTTPException(status_code=422, detail="portion_g must be greater than zero")

    top_k = max(1, min(int(k), MAX_CLASSES))

    try:
        content = await image.read()
        if not content:
            raise ValueError("Empty image payload")

        x = prep_bytes(content)
        probs = model.predict(x, verbose=0)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    idxs = np.argsort(probs)[-top_k:][::-1].astype(int)
    top = [FoodTopK(label=LABELS[i], score=float(probs[i])) for i in idxs]
    best = top[0].label

    return FoodResponse(
        best_label=best,
        topk=top,
        nutrition_per_portion=build_nutrition(best, portion_g),
    )
