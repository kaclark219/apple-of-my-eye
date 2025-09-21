# api/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib, numpy as np, cv2
from PIL import Image, ImageOps
from io import BytesIO
from src.features import extract_features_from_bgr

app = FastAPI(title="apple (?) of my eye")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.joblib"
clf = joblib.load(MODEL_PATH)
CLASSES = clf.classes_

def preprocess_to_100_bgr(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = ImageOps.exif_transpose(img)

    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top  = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))

    img = img.resize((100, 100), Image.Resampling.BOX)

    arr = np.array(img)
    bgr = arr[:, :, ::-1].copy()
    return bgr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    bgr100 = preprocess_to_100_bgr(data)
    x = extract_features_from_bgr(bgr100).reshape(1, -1)
    probs = clf.predict_proba(x)[0]
    i = int(np.argmax(probs))
    top3 = sorted(
        [{"label": CLASSES[j], "p": float(p)} for j, p in enumerate(probs)],
        key=lambda z: z["p"], reverse=True
    )[:3]
    return {"label": CLASSES[i], "confidence": float(probs[i]), "top3": top3}