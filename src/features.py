import cv2
import numpy as np
from skimage.feature import hog
from pathlib import Path

def _color_hist_rgb(img_rgb: np.ndarray, bins: int = 16) -> np.ndarray:
    hists = []
    for i in range(3):
        hist = cv2.calcHist([img_rgb], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
        hists.append(hist)
    return np.hstack(hists)

def _hog_gray(gray: np.ndarray) -> np.ndarray:
    return hog(
        gray,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        orientations=9,
        feature_vector=True,
    )

def extract_features_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("extract_features_from_bgr: got None image")

    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    img_bgr = img_bgr[y0:y0 + s, x0:x0 + s]

    if img_bgr.shape[:2] != (100, 100):
        img_bgr = cv2.resize(img_bgr, (100, 100), interpolation=cv2.INTER_AREA)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    color_hist = _color_hist_rgb(img_rgb, bins=16)
    hog_feat = _hog_gray(gray)

    return np.hstack([color_hist, hog_feat]).astype(np.float32)

def extract_features(path) -> np.ndarray:
    """
    Path-based wrapper used by training scripts that read from disk.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return extract_features_from_bgr(img_bgr)