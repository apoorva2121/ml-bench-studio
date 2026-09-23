"""M4 forecasting — Naive2 reference plus the statistical baselines, on any series you paste."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from registry import ML_BENCH, Example, Field, Plugin, parse_numbers

REPO = ML_BENCH / "m4-forecasting"
sys.path.insert(0, str(REPO / "src"))

# Box & Jenkins airline passengers (monthly, 1949-1960) — public domain classic.
AIR = ("112 118 132 129 121 135 148 148 136 119 104 118 115 126 141 135 125 149 170 170 158 133 114 140 "
       "145 150 178 163 172 178 199 199 184 162 146 166 171 180 193 181 183 218 230 242 209 191 172 194 "
       "196 196 236 235 229 243 264 272 237 211 180 201 204 188 235 227 234 264 302 293 259 229 203 229 "
       "242 233 267 269 270 315 364 347 312 274 237 278 284 277 317 313 318 374 413 405 355 306 271 306 "
       "315 301 356 348 355 422 465 467 404 347 305 336 340 318 362 348 363 435 491 505 404 359 310 337 "
       "360 342 406 396 420 472 548 559 463 407 362 405 417 391 419 461 472 535 622 606 508 461 390 432")


def _m4_example() -> Example | None:
    p = REPO / "data" / "Quarterly-train.csv"
    if not p.exists():
        return None
    try:
        row = pd.read_csv(p, nrows=5).iloc[2]
        y = row.drop(labels=[row.index[0]]).dropna().astype(float).tolist()
        return Example(f"M4 quarterly series {row.iloc[0]}", {"series": " ".join(f"{x:g}" for x in y[-60:]), "horizon": 8, "m": 4, "holdout": True})
    except Exception:
        return None


def run(v: dict) -> dict:
    from m4_forecasting.baselines import naive2_forecast
    from m4_forecasting.metrics import mase, smape
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, SeasonalNaive, Theta

    y_all = np.array(parse_numbers(v["series"]), dtype=float)
    h, m = int(v["horizon"]), int(v["m"])
    if len(y_all) < max(2 * m, 8) + (h if v.get("holdout") else 0):
        raise ValueError("series too short for this horizon / seasonality")
    if v.get("holdout"):
        y, y_test = y_all[:-h], y_all[-h:]
    else:
        y, y_test = y_all, None

    fc = {"Naive2": naive2_forecast(y, h, m)}
    df = pd.DataFrame({"unique_id": "s", "ds": np.arange(len(y)), "y": y})
    sf = StatsForecast(models=[Theta(season_length=m), SeasonalNaive(season_length=m), AutoETS(season_length=m)], freq=1, n_jobs=1)
    out = sf.forecast(df=df, h=h)
    for col in ("Theta", "SeasonalNaive", "AutoETS"):
        if col in out:
            fc[col] = out[col].to_numpy(dtype=float)
    fc["Ensemble"] = np.mean([fc[k] for k in ("Theta", "AutoETS", "SeasonalNaive") if k in fc], axis=0)

    metrics, table = [], None
    if y_test is not None:
        rows = []
        for k, f in fc.items():
            rows.append({"method": k, "sMAPE": round(smape(y_test, f), 3), "MASE": round(mase(y_test, f, y, m), 3)})
        table = {"columns": ["method", "sMAPE", "MASE"], "rows": rows}
        best = min(rows, key=lambda r: r["sMAPE"])
        metrics = [{"label": "Best sMAPE", "value": f"{best['sMAPE']}", "hint": best["method"]},
                   {"label": "Naive2 sMAPE", "value": f"{[r for r in rows if r['method']=='Naive2'][0]['sMAPE']}", "hint": "the M4 reference"},
                   {"label": "Held-out points", "value": str(h)}]
    x = list(range(len(y_all) if y_test is not None else len(y) + h))
    series = [{"name": "history", "y": y.tolist() + [None] * h}]
    if y_test is not None:
        series.append({"name": "actual (held out)", "y": [None] * len(y) + y_test.tolist()})
    for k, f in fc.items():
        series.append({"name": k, "y": [None] * len(y) + [float(z) for z in f]})
    return {"summary": f"{len(fc)} forecasts, horizon {h}, seasonality {m}" + (" — scored on the held-out tail" if y_test is not None else ""),
            "metrics": metrics, "table": table, "series": {"x": x, "series": series, "y_label": "value"},
            "details": {k: [round(float(z), 2) for z in f] for k, f in fc.items()}}


ex = [Example("Airline passengers (monthly)", {"series": AIR, "horizon": 12, "m": 12, "holdout": True},
              note="Last 12 months held out and scored"),
      Example("Linear trend + noise", {"series": " ".join(f"{100 + 2 * i + ((i * 7) % 5 - 2):g}" for i in range(48)), "horizon": 6, "m": 1, "holdout": True})]
m4ex = _m4_example()
if m4ex:
    ex.insert(1, m4ex)

PLUGIN = Plugin(
    id="m4-forecasting", title="M4 Forecasting Baselines", tagline="Naive2, Theta, ETS, seasonal naive and their ensemble",
    category="Time series", repo="m4-forecasting", headline_metric="OWA 0.858 on M4 (Ensemble)", icon="trend",
    description="Paste any series. The plugin fits the same baselines used in the M4 benchmark, optionally holds out the last "
                "`horizon` points and scores each method with sMAPE and MASE exactly as the competition did.",
    fields=[Field("series", "Series values", "textarea", AIR, help="Numbers separated by spaces or commas, oldest first"),
            Field("horizon", "Forecast horizon", "number", 12, min=1, max=48, step=1),
            Field("m", "Seasonal period", "number", 12, min=1, max=52, step=1, help="12 monthly, 4 quarterly, 1 none"),
            Field("holdout", "Hold out last `horizon` points and score", "checkbox", True, required=False)],
    examples=ex, run=run,
)
