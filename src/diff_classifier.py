"""
Загрузка классификатора diff: PyTorch (.pt) или scikit-learn (.joblib) без DLL torch.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image

ClassifierHandle = Tuple[str, Any]
# kind: "torch" | "sklearn"


def diff_gray_vector(diff_path: str, size: int = 64) -> np.ndarray:
    im = Image.open(diff_path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return (np.asarray(im, dtype=np.float32).flatten() / 255.0).astype(np.float32)


def load_classifier(path: str | None) -> Tuple[Optional[ClassifierHandle], bool]:
    if not path or not os.path.isfile(path):
        return None, False
    low = path.lower()
    if low.endswith(".joblib"):
        return _load_sklearn(path)
    # .pt; при битом torch — соседний *_sklearn.joblib
    h, ok = _load_torch(path)
    if ok:
        return h, True
    alt = path.replace(".pt", "_sklearn.joblib")
    if alt != path and os.path.isfile(alt):
        return _load_sklearn(alt)
    return None, False


def _load_sklearn(path: str) -> Tuple[Optional[ClassifierHandle], bool]:
    try:
        import joblib
    except ImportError:
        return None, False
    try:
        blob = joblib.load(path)
        if not isinstance(blob, dict) or "clf" not in blob:
            return None, False
        return ("sklearn", blob), True
    except Exception:
        return None, False


def _load_torch(path: str) -> Tuple[Optional[ClassifierHandle], bool]:
    try:
        import torch

        from src.model_net import TinyDiffCNN
    except OSError:
        return None, False
    device = torch.device("cpu")
    m = TinyDiffCNN()
    try:
        try:
            state = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=device)
        m.load_state_dict(state)
        m.to(device)
        m.eval()
        return ("torch", (m, device)), True
    except Exception:
        return None, False


def predict_fail_prob(handle: ClassifierHandle, diff_path: str) -> float:
    kind, payload = handle
    if kind == "sklearn":
        blob = payload
        clf = blob["clf"]
        size = int(blob.get("size", 64))
        vec = diff_gray_vector(diff_path, size=size).reshape(1, -1)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(vec)
            return float(proba[0, 1])
        pred = clf.predict(vec)
        return float(pred[0])
    m, device = payload
    import torch

    from src.compare import diff_tensor_gray

    x = diff_tensor_gray(diff_path)
    with torch.no_grad():
        logits = m(x.to(device))
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)
