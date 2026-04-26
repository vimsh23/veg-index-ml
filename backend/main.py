"""
Vegetation Index ML API - FastAPI Backend
Supports: GeoTIFF upload, CSV upload, NDVI/EVI/SAVI/NDWI computation, RF prediction
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vegetation Index ML API",
    description="Remote sensing vegetation index computation and land cover classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/rf_model.joblib")
model = None
label_encoder = None
feature_names = None

@app.on_event("startup")
def load_model():
    global model, label_encoder, feature_names
    if os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        # Support both dict format (from train.py) and raw model format
        if isinstance(payload, dict):
            model = payload["model"]
            label_encoder = payload.get("label_encoder")
            feature_names = payload.get("feature_names")
        else:
            model = payload
        logger.info(f"Model loaded from {MODEL_PATH} | classes: {model.classes_}")
    else:
        logger.warning(f"No model found at {MODEL_PATH}. Run: python model/train.py")


# ─── Index formulas ─────────────────────────────────────────────────────────

def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)

def compute_evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L + 1e-10)

def compute_savi(nir, red, L=0.5):
    return ((nir - red) / (nir + red + L + 1e-10)) * (1 + L)

def compute_ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-10)

def compute_msavi(nir, red):
    return (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2

INDEX_REGISTRY = {
    "NDVI":  lambda b: compute_ndvi(b["nir"], b["red"]),
    "EVI":   lambda b: compute_evi(b["nir"], b["red"], b["blue"]),
    "SAVI":  lambda b: compute_savi(b["nir"], b["red"]),
    "NDWI":  lambda b: compute_ndwi(b["green"], b["nir"]),
    "MSAVI": lambda b: compute_msavi(b["nir"], b["red"]),
}

BAND_NAMES = ["blue", "green", "red", "nir"]


def read_geotiff_bands(contents: bytes) -> dict:
    """Parse GeoTIFF and return band dict. Assumes Sentinel-2 band order: B2,G3,R4,NIR8."""
    try:
        import rasterio
        with rasterio.open(io.BytesIO(contents)) as src:
            scale = 10000.0
            n = src.count
            bands = {}
            mapping = {1: "blue", 2: "green", 3: "red", 4: "nir"}
            for band_idx, name in mapping.items():
                if band_idx <= n:
                    bands[name] = src.read(band_idx).astype(np.float32) / scale
            return bands
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"GeoTIFF read error: {str(e)}")


def read_csv_bands(contents: bytes) -> dict:
    """Parse CSV with columns: blue, green, red, nir (reflectance 0-1 or DN)."""
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = [c.strip().lower() for c in df.columns]
        bands = {}
        for name in BAND_NAMES:
            if name in df.columns:
                arr = df[name].values.astype(np.float32)
                # Auto-scale if DN (values > 1 suggest DN not reflectance)
                if arr.max() > 1.0:
                    arr = arr / 10000.0
                bands[name] = arr
        return bands, df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV read error: {str(e)}")


def band_stats(arr: np.ndarray) -> dict:
    flat = arr.flatten()
    flat = flat[~np.isnan(flat)]
    return {
        "min": round(float(np.min(flat)), 4),
        "max": round(float(np.max(flat)), 4),
        "mean": round(float(np.mean(flat)), 4),
        "std": round(float(np.std(flat)), 4),
        "percentile_25": round(float(np.percentile(flat, 25)), 4),
        "percentile_75": round(float(np.percentile(flat, 75)), 4),
    }


# ─── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Vegetation Index ML API is running.", "docs": "/docs"}


@app.get("/indices")
def list_indices():
    return {"available_indices": list(INDEX_REGISTRY.keys())}


@app.post("/compute-index/geotiff")
async def compute_index_geotiff(
    file: UploadFile = File(...),
    index: str = Query("NDVI", enum=list(INDEX_REGISTRY.keys()))
):
    """Compute vegetation index from a GeoTIFF file."""
    contents = await file.read()
    bands = read_geotiff_bands(contents)

    if index not in INDEX_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown index: {index}")

    result = INDEX_REGISTRY[index](bands)
    flat = result.flatten()
    sample = flat[~np.isnan(flat)][:1000].tolist()

    return {
        "index": index,
        "stats": band_stats(result),
        "shape": list(result.shape),
        "sample_values": sample,
        "histogram": np.histogram(sample, bins=20)[0].tolist(),
        "histogram_edges": np.histogram(sample, bins=20)[1].tolist(),
    }


@app.post("/compute-index/csv")
async def compute_index_csv(
    file: UploadFile = File(...),
    index: str = Query("NDVI", enum=list(INDEX_REGISTRY.keys()))
):
    """Compute vegetation index from a CSV with band columns."""
    contents = await file.read()
    bands, df = read_csv_bands(contents)

    if index not in INDEX_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown index: {index}")

    result = INDEX_REGISTRY[index](bands)
    sample = result[:1000].tolist()

    return {
        "index": index,
        "stats": band_stats(result),
        "n_pixels": int(len(result)),
        "sample_values": sample,
        "histogram": np.histogram(result, bins=20)[0].tolist(),
        "histogram_edges": np.histogram(result, bins=20)[1].tolist(),
    }


@app.post("/compute-all/csv")
async def compute_all_indices_csv(file: UploadFile = File(...)):
    """Compute all vegetation indices from a CSV and return a summary."""
    contents = await file.read()
    bands, _ = read_csv_bands(contents)
    results = {}
    for name, fn in INDEX_REGISTRY.items():
        try:
            arr = fn(bands)
            results[name] = band_stats(arr)
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"indices": results}


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predict land cover class from CSV band data using trained RF model."""
    global model, label_encoder
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python model/train.py")

    contents = await file.read()
    bands, df = read_csv_bands(contents)

    # Build feature matrix: raw bands first, then indices (must match train.py order)
    features = []
    for name in BAND_NAMES:
        if name in bands:
            features.append(bands[name])
        else:
            raise HTTPException(status_code=400, detail=f"Missing band column: '{name}'. CSV needs: blue, green, red, nir")

    for idx_name, fn in INDEX_REGISTRY.items():
        try:
            features.append(fn(bands))
        except Exception:
            features.append(np.zeros(len(list(bands.values())[0])))

    X = np.stack(features, axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    preds = model.predict(X)

    # Decode labels if label encoder available
    CLASS_NAMES_MAP = {0: "Water", 1: "Forest", 2: "Agriculture", 3: "Urban", 4: "Bare Soil"}
    unique, counts = np.unique(preds, return_counts=True)

    dist = {}
    for k, v in zip(unique, counts):
        k_int = int(k)
        if label_encoder is not None:
            try:
                label = str(label_encoder.inverse_transform([k_int])[0])
            except Exception:
                label = CLASS_NAMES_MAP.get(k_int, f"Class {k_int}")
        else:
            label = CLASS_NAMES_MAP.get(k_int, f"Class {k_int}")
        dist[label] = int(v)

    # Feature importance top 5
    importances = []
    if hasattr(model, "feature_importances_") and feature_names:
        fi = list(zip(feature_names, model.feature_importances_))
        fi.sort(key=lambda x: -x[1])
        importances = [{"feature": f, "importance": round(float(i), 4)} for f, i in fi[:5]]

    return {
        "n_samples": int(len(preds)),
        "class_distribution": dist,
        "feature_importance": importances,
        "classes": model.classes_.tolist() if hasattr(model, "classes_") else [],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
