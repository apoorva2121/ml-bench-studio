"""ML serving template — the validated 7-feature model behind the FastAPI example service."""

from __future__ import annotations

import json
import sys
from functools import lru_cache

from registry import ML_BENCH, Example, Field, Plugin

REPO = ML_BENCH / "ml-serving-template"
sys.path.insert(0, str(REPO / "src"))


@lru_cache(maxsize=1)
def _pred():
    from serving.predictor import Predictor
    return Predictor(REPO / "artifacts" / "model.joblib")


def run(v: dict) -> dict:
    from serving.schema import HouseFeatures          # the same pydantic validation the API applies
    row = HouseFeatures(**{k: v[k] for k in ("OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt", "FullBath", "Neighborhood")}).model_dump()
    p = _pred()
    price = float(p.predict([row])[0])
    card = json.loads((REPO / "artifacts" / "model_card.json").read_text()) if (REPO / "artifacts" / "model_card.json").exists() else {}
    return {"summary": f"Predicted sale price ${price:,.0f}",
            "metrics": [{"label": "Predicted price", "value": f"${price:,.0f}"},
                        {"label": "Model version", "value": p.version},
                        {"label": "CV RMSLE", "value": f"{card.get('cv_rmsle', float('nan')):.3f}", "hint": "7 features only — see Ames plugin for the full model"}],
            "details": {"validated_request": row, "model_card": card}}


PLUGIN = Plugin(
    id="ml-serving-template", title="Serving Template Model", tagline="Pydantic-validated request → HistGradientBoosting → price",
    category="Serving", repo="ml-serving-template", headline_metric="CV RMSLE 0.161 · p95 latency budget enforced in CI", icon="server",
    description="The small model that ships with my FastAPI/MLflow/Docker serving template. Same validation as the live "
                "endpoint: out-of-range values are rejected before the model sees them.",
    fields=[Field("OverallQual", "Overall quality (1-10)", "number", 7, min=1, max=10, step=1),
            Field("GrLivArea", "Living area (sq ft)", "number", 1800, min=1, max=10000),
            Field("TotalBsmtSF", "Basement (sq ft)", "number", 1000, min=0, max=10000),
            Field("GarageCars", "Garage cars", "number", 2, min=0, max=5, step=1),
            Field("YearBuilt", "Year built", "number", 2003, min=1800, max=2030, step=1),
            Field("FullBath", "Full baths", "number", 2, min=0, max=6, step=1),
            Field("Neighborhood", "Neighbourhood", "select", "CollgCr", options=["CollgCr", "NAmes", "OldTown", "NridgHt", "Somerst", "Edwards", "Gilbert", "StoneBr", "BrkSide"])],
    examples=[Example("Newer suburban", {"OverallQual": 7, "GrLivArea": 1800, "TotalBsmtSF": 1000, "GarageCars": 2, "YearBuilt": 2003, "FullBath": 2, "Neighborhood": "CollgCr"}),
              Example("Premium NridgHt", {"OverallQual": 9, "GrLivArea": 2600, "TotalBsmtSF": 1800, "GarageCars": 3, "YearBuilt": 2007, "FullBath": 3, "Neighborhood": "NridgHt"}),
              Example("Old town fixer", {"OverallQual": 4, "GrLivArea": 1100, "TotalBsmtSF": 600, "GarageCars": 1, "YearBuilt": 1925, "FullBath": 1, "Neighborhood": "OldTown"})],
    run=run, status="ready" if (REPO / "artifacts" / "model.joblib").exists() else "needs-artifacts",
)
