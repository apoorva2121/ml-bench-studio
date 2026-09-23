"""MedMNIST — classify a test image with the trained CNNs and show where the model looked (Grad-CAM)."""

from __future__ import annotations

import base64
import io
import os
import sys
from functools import lru_cache

import numpy as np

from registry import ML_BENCH, Example, Field, Plugin

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
REPO = ML_BENCH / "medmnist-classification"
sys.path.insert(0, str(REPO / "src"))
DATASETS = [d for d in ("pneumoniamnist", "dermamnist") if (REPO / "data" / f"{d}.npz").exists()]
MODELS = ["small_cnn", "resnet18_28"]
LABELS = {"pneumoniamnist": {0: "normal", 1: "pneumonia"},
          "dermamnist": {0: "actinic keratoses", 1: "basal cell carcinoma", 2: "benign keratosis", 3: "dermatofibroma",
                         4: "melanoma", 5: "melanocytic nevi", 6: "vascular lesions"}}


@lru_cache(maxsize=4)
def _data(ds: str):
    d = np.load(REPO / "data" / f"{ds}.npz")
    tr = d["train_images"].astype("float32") / 255.0
    return d["test_images"], d["test_labels"].ravel(), float(tr.mean()), float(tr.std())


@lru_cache(maxsize=4)
def _model(ds: str, name: str):
    import torch
    from medcls.models import make_model
    x, _, _, _ = _data(ds)
    in_ch = 1 if x.ndim == 3 else x.shape[-1]
    m = make_model(name, in_ch, len(LABELS[ds]))
    state = torch.load(REPO / "artifacts" / f"{ds}_{name}.pt", map_location="cpu", weights_only=True)  # state_dict only
    m.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    return m.eval()


def _png(arr: np.ndarray, scale: int = 6, cmap: np.ndarray | None = None) -> str:
    from PIL import Image
    if arr.ndim == 2:
        img = Image.fromarray((arr * 255).clip(0, 255).astype("uint8"), "L").convert("RGB")
    else:
        img = Image.fromarray((arr * 255).clip(0, 255).astype("uint8"), "RGB")
    if cmap is not None:
        heat = Image.fromarray((np.stack([cmap, np.zeros_like(cmap), 1 - cmap], -1) * 255).astype("uint8"), "RGB")
        img = Image.blend(img, heat, 0.45)
    img = img.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run(v: dict) -> dict:
    import torch
    from medcls.evaluate import grad_cam
    ds, name, i = v["dataset"], v["model"], int(v["index"])
    x_test, y_test, mean, std = _data(ds)
    if i >= len(x_test):
        raise ValueError(f"index must be < {len(x_test)}")
    img = x_test[i].astype("float32") / 255.0
    xt = torch.tensor((img - mean) / std)
    xt = xt.unsqueeze(0) if xt.ndim == 2 else xt.permute(2, 0, 1)
    xt = xt.unsqueeze(0)
    m = _model(ds, name)
    with torch.no_grad():
        probs = torch.softmax(m(xt), 1)[0].numpy()
    pred = int(probs.argmax())
    cam = grad_cam(m, xt, pred)
    cam = np.asarray(cam, dtype="float32")
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    if cam.shape != img.shape[:2]:
        from PIL import Image
        cam = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(img.shape[:2][::-1], Image.BILINEAR)) / 255.0
    lab = LABELS[ds]
    truth = int(y_test[i])
    return {"summary": f"Predicted {lab[pred]} ({probs[pred]:.0%}) — actual {lab[truth]} → {'correct' if pred == truth else 'WRONG'}",
            "metrics": [{"label": "Prediction", "value": lab[pred], "hint": f"{probs[pred]:.1%} confidence"},
                        {"label": "Ground truth", "value": lab[truth]},
                        {"label": "Model", "value": name}],
            "bars": {"labels": [lab[k] for k in range(len(lab))], "values": [float(p) for p in probs], "label": "Class probabilities"},
            "image": _png(img, cmap=cam), "image_caption": "Test image with Grad-CAM overlay (red = high attention)",
            "image_plain": _png(img),
            "details": {"index": i, "probabilities": {lab[k]: round(float(p), 4) for k, p in enumerate(probs)}}}


PLUGIN = Plugin(
    id="medmnist", title="MedMNIST Image Classification", tagline="Small CNN and ResNet-18 with Grad-CAM, trained on CPU",
    category="Vision", repo="medmnist-classification", headline_metric="AUC 0.979 Pneumonia · 0.933 Derma (small CNN)", icon="image",
    description="Pick a test image by index, choose a model, and see the prediction, its confidence, and a Grad-CAM heat-map of "
                "where the network looked. Test images were never used in training or model selection.",
    fields=[Field("dataset", "Dataset", "select", DATASETS[0] if DATASETS else "pneumoniamnist", options=DATASETS or ["pneumoniamnist"]),
            Field("model", "Model", "select", "small_cnn", options=MODELS),
            Field("index", "Test image index", "number", 0, min=0, max=2004, step=1)],
    examples=[Example("Chest X-ray #3, small CNN", {"dataset": "pneumoniamnist", "model": "small_cnn", "index": 3}),
              Example("Chest X-ray #500, ResNet-18", {"dataset": "pneumoniamnist", "model": "resnet18_28", "index": 500}),
              Example("Dermatoscope #42, small CNN", {"dataset": "dermamnist", "model": "small_cnn", "index": 42}),
              Example("Dermatoscope #1200, ResNet-18", {"dataset": "dermamnist", "model": "resnet18_28", "index": 1200})],
    run=run, heavy=True,
    status="ready" if DATASETS and (REPO / "artifacts").exists() else "needs-artifacts",
)
