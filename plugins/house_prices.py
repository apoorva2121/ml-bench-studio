"""Ames house prices — the blended GBDT/linear model from ml-bench/house-prices."""

from __future__ import annotations

import sys
from functools import lru_cache

import pandas as pd

from registry import ML_BENCH, Example, Field, Plugin

REPO = ML_BENCH / "house-prices"
sys.path.insert(0, str(REPO / "src"))

EDITABLE = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt", "YearRemodAdd",
            "FullBath", "LotArea", "Neighborhood", "KitchenQual"]


@lru_cache(maxsize=1)
def _load():
    import joblib
    bundle = joblib.load(REPO / "artifacts" / "model.joblib")   # produced by our own train.py only
    train = pd.read_csv(REPO / "data" / "train.csv")
    return bundle["builder"], bundle["blend"], train


def _examples() -> list[Example]:
    try:
        _, _, train = _load()
    except Exception:
        return []
    picks = {1: "Typical 2-storey, NAmes-like", 692: "Luxury, NoRidge (top-5 price)", 30: "Small older house, BrkSide"}
    ex = []
    for i, label in picks.items():
        row = train[train["Id"] == i]
        if row.empty:
            continue
        r = row.iloc[0]
        vals = {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in EDITABLE}
        vals["base_id"] = int(i)
        ex.append(Example(label, vals, note=f"Actual SalePrice ${r['SalePrice']:,.0f} (training row — the model has seen it)"))
    return ex


def run(v: dict) -> dict:
    from house_prices.features import engineer
    from house_prices.preprocessing import clean
    builder, blend, train = _load()
    base = train[train["Id"] == int(v.get("base_id") or 1)]
    if base.empty:
        base = train.iloc[[0]]
    row = base.drop(columns=["SalePrice"]).copy()
    for c in EDITABLE:
        if c in v and v[c] not in (None, ""):
            row[c] = int(v[c]) if train[c].dtype.kind in "iu" else v[c]
    X = builder.transform(engineer(clean(row.drop(columns=["Id"]))))
    pred = float(blend.predict(X)[0])
    actual = float(base["SalePrice"].iloc[0])
    per_model = {}
    if hasattr(blend, "models_") and hasattr(blend, "weights_"):
        try:
            import numpy as np
            for name, m in blend.models_.items():
                per_model[name] = float(np.expm1(m.predict(X)[0])) if getattr(blend, "log_target", True) else float(m.predict(X)[0])
        except Exception:
            per_model = {}
    return {
        "summary": f"Predicted sale price ${pred:,.0f}",
        "metrics": [{"label": "Predicted price", "value": f"${pred:,.0f}"},
                    {"label": "Actual (base row)", "value": f"${actual:,.0f}", "hint": "unchanged only if you kept the example values"},
                    {"label": "Difference", "value": f"{(pred - actual) / actual:+.1%}"}],
        "bars": {"labels": list(per_model) or ["blend"], "values": list(per_model.values()) or [pred], "label": "Per-model prediction ($)"} if per_model else None,
        "details": {"edited_fields": {c: v.get(c) for c in EDITABLE}, "base_id": v.get("base_id")},
        "warnings": ["Fields not shown are taken from the chosen base row (all 79 Ames features are needed)."],
    }


PLUGIN = Plugin(
    id="house-prices", title="Ames House Prices", tagline="Blended GBDT + regularised linear regression",
    category="Tabular", repo="house-prices", headline_metric="OOF RMSLE 0.108", icon="home",
    description="Edit the key features of a real Ames house and see the blended model re-price it. "
                "The full 79-feature row comes from the chosen base example; you override the ones that matter most.",
    fields=[
        Field("base_id", "Base row (train Id)", "number", 1, min=1, max=1460, step=1, help="Which training house supplies the other 69 features"),
        Field("OverallQual", "Overall quality (1-10)", "number", 6, min=1, max=10, step=1),
        Field("GrLivArea", "Above-ground living area (sq ft)", "number", 1500, min=300, max=6000, step=1),
        Field("TotalBsmtSF", "Basement area (sq ft)", "number", 900, min=0, max=6000, step=1),
        Field("GarageCars", "Garage capacity (cars)", "number", 2, min=0, max=4, step=1),
        Field("YearBuilt", "Year built", "number", 1990, min=1870, max=2010, step=1),
        Field("YearRemodAdd", "Year remodelled", "number", 1995, min=1950, max=2010, step=1),
        Field("FullBath", "Full bathrooms", "number", 2, min=0, max=4, step=1),
        Field("LotArea", "Lot area (sq ft)", "number", 9000, min=1000, max=100000, step=1),
        Field("Neighborhood", "Neighbourhood", "select", "NAmes",
              options=["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "Gilbert", "NridgHt", "Sawyer", "NWAmes",
                       "SawyerW", "BrkSide", "Crawfor", "Mitchel", "NoRidge", "Timber", "IDOTRR", "ClearCr", "StoneBr",
                       "SWISU", "MeadowV", "Blmngtn", "BrDale", "Veenker", "NPkVill", "Blueste"]),
        Field("KitchenQual", "Kitchen quality", "select", "TA", options=["Ex", "Gd", "TA", "Fa"]),
    ],
    examples=_examples(),
    run=run,
    status="ready" if (REPO / "artifacts" / "model.joblib").exists() else "needs-artifacts",
)
