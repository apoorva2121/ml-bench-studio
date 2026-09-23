"""Bayesian marketing-mix model — response curves from the fitted posterior, plus a spend what-if."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from registry import ML_BENCH, Example, Field, Plugin

REPO = ML_BENCH / "bayesian-mmm"
sys.path.insert(0, str(REPO / "src"))
REPORTS = REPO / "reports"


def _posterior() -> pd.DataFrame | None:
    p = REPORTS / "parameter_recovery.csv"
    return pd.read_csv(p) if p.exists() else None


def run(v: dict) -> dict:
    from mmm.transforms import geometric_adstock_np, hill_np
    ch = v["channel"]
    post = _posterior()
    params = {"adstock_decay": v["decay"], "hill_half_sat": v["half_sat"], "hill_slope": v["slope"]}
    source = "your inputs"
    if v.get("use_posterior") and post is not None:
        sub = post[post["channel"] == ch].set_index("parameter")["posterior_median"]
        params = {k: float(sub[k]) for k in params if k in sub}
        source = "posterior median from the fitted model"
    weeks = int(v["weeks"])
    spend = np.zeros(weeks)
    spend[: int(v["burst_weeks"])] = v["weekly_spend"]           # a burst, then silence -> shows carry-over
    adstocked = geometric_adstock_np(spend, params["adstock_decay"], int(v["max_lag"]))
    response = hill_np(adstocked, params["hill_half_sat"], params["hill_slope"])
    grid = np.linspace(0, 3 * params["hill_half_sat"], 60)
    curve = hill_np(grid, params["hill_half_sat"], params["hill_slope"])
    marginal = np.gradient(curve, grid)
    cur = hill_np(np.array([v["weekly_spend"]]), params["hill_half_sat"], params["hill_slope"])[0]
    table = None
    if post is not None:
        rows = post[post["channel"] == ch][["parameter", "true", "posterior_median", "hdi_94_low", "hdi_94_high", "recovered"]]
        table = {"columns": list(rows.columns), "rows": [{k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()} for r in rows.to_dict("records")]}
    return {
        "summary": f"{ch}: saturation at your weekly spend is {cur:.0%} of maximum ({source})",
        "metrics": [{"label": "Adstock decay", "value": f"{params['adstock_decay']:.2f}", "hint": "share of effect carried to next week"},
                    {"label": "Half-saturation spend", "value": f"{params['hill_half_sat']:,.0f}"},
                    {"label": "Saturation at your spend", "value": f"{cur:.0%}"},
                    {"label": "Weeks to 90% decay", "value": f"{int(np.ceil(np.log(0.1) / np.log(max(params['adstock_decay'], 1e-6))))}"}],
        "series": {"x": list(range(weeks)), "y_label": "normalised",
                   "series": [{"name": "spend (scaled)", "y": (spend / max(spend.max(), 1)).tolist()},
                              {"name": "adstocked (scaled)", "y": (adstocked / max(adstocked.max(), 1e-9)).tolist()},
                              {"name": "saturated response", "y": response.tolist()}]},
        "bars": {"labels": [f"{g:,.0f}" for g in grid[::6]], "values": [float(c) for c in curve[::6]], "label": "Response vs weekly spend (Hill curve)"},
        "table": table,
        "details": {"params_used": params, "marginal_response_at_spend": float(np.interp(v["weekly_spend"], grid, marginal))},
        "warnings": [] if post is not None else ["reports/parameter_recovery.csv not found — using your inputs only"],
    }


post = _posterior()
channels = sorted(post["channel"].unique().tolist()) if post is not None else ["tv", "search", "social", "display"]
PLUGIN = Plugin(
    id="bayesian-mmm", title="Bayesian Marketing Mix", tagline="Adstock carry-over and Hill saturation, recovered with PyMC",
    category="Causal", repo="bayesian-mmm", headline_metric="11/12 parameters recovered in 94% HDI", icon="layers",
    description="Explore what the fitted model believes about a channel: how long spend keeps working (adstock) and where it "
                "stops paying back (saturation). Toggle off the posterior to try your own parameters.",
    fields=[Field("channel", "Channel", "select", channels[0], options=channels),
            Field("use_posterior", "Use fitted posterior medians", "checkbox", True, required=False),
            Field("weekly_spend", "Weekly spend", "number", 50000, min=0, max=1_000_000),
            Field("burst_weeks", "Weeks of spend before stopping", "number", 4, min=1, max=26, step=1),
            Field("weeks", "Weeks to simulate", "number", 16, min=4, max=52, step=1),
            Field("max_lag", "Max adstock lag (weeks)", "number", 8, min=1, max=13, step=1),
            Field("decay", "Adstock decay (manual)", "number", 0.5, min=0, max=0.99, step=0.01),
            Field("half_sat", "Half-saturation spend (manual)", "number", 60000, min=1, max=1_000_000),
            Field("slope", "Hill slope (manual)", "number", 1.5, min=0.1, max=5, step=0.1)],
    examples=[Example("TV: long carry-over", {"channel": "tv", "use_posterior": True, "weekly_spend": 54000, "burst_weeks": 4, "weeks": 16, "max_lag": 8}),
              Example("Search: fast decay, high ROI", {"channel": "search", "use_posterior": True, "weekly_spend": 23000, "burst_weeks": 4, "weeks": 12, "max_lag": 8}),
              Example("Manual: over-saturated channel", {"channel": channels[0], "use_posterior": False, "weekly_spend": 150000, "decay": 0.3, "half_sat": 40000, "slope": 2.0, "burst_weeks": 3, "weeks": 12, "max_lag": 6})],
    run=run,
)
