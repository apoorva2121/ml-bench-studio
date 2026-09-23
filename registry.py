"""Plugin contract and discovery for ML Bench Studio.

Adding a model = dropping one file into `plugins/` that defines a module-level `PLUGIN`.
Nothing else changes: the home grid, the model screen, the examples and the result
renderers are all driven by this contract.

Result dict keys the UI knows how to render (all optional):
  summary   str                    one-line headline
  metrics   [{label, value, hint}] big-number cards
  table     {columns, rows}        tabular output
  series    {x, series:[{name, y}], y_label}   line chart
  bars      {labels, values, label}            bar chart
  image     data-URL               e.g. Grad-CAM overlay
  text      str                    long-form text (answers, notes)
  citations [{n, source, url, text}]
  details   dict                   collapsible JSON (traces, raw output)
  warnings  [str]
"""

from __future__ import annotations

import importlib
import pkgutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
ML_BENCH = ROOT.parent


@dataclass
class Field:
    name: str
    label: str
    type: str = "number"                   # number | text | textarea | select | password | checkbox
    default: Any = None
    options: list[str] | None = None
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    required: bool = True


@dataclass
class Example:
    label: str
    values: dict[str, Any]
    note: str = ""


@dataclass
class Plugin:
    id: str
    title: str
    tagline: str
    category: str                          # Tabular | Time series | Causal | Serving | Retrieval | Vision | GenAI
    repo: str                              # directory name under ml-bench
    headline_metric: str
    fields: list[Field]
    examples: list[Example]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ""
    heavy: bool = False                    # first call loads big models; UI shows a longer loading state
    status: str = "ready"                  # ready | needs-artifacts | needs-key
    icon: str = "cube"                     # name of an inline SVG icon in the UI

    def public(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "run"}
        d["repo_path"] = str(ML_BENCH / self.repo)
        return d


class Registry:
    def __init__(self) -> None:
        self.plugins: dict[str, Plugin] = {}
        self.load_errors: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self.discover()

    def discover(self) -> None:
        pkg = importlib.import_module("plugins")
        for m in pkgutil.iter_modules(pkg.__path__):
            try:
                mod = importlib.import_module(f"plugins.{m.name}")
                p: Plugin = getattr(mod, "PLUGIN")
                self.plugins[p.id] = p
                self._locks[p.id] = threading.Lock()
            except Exception as e:  # a broken plugin must not take the studio down
                self.load_errors[m.name] = f"{type(e).__name__}: {e}"

    def run(self, plugin_id: str, values: dict[str, Any]) -> dict[str, Any]:
        p = self.plugins[plugin_id]
        clean = coerce(p.fields, values)
        t0 = time.perf_counter()
        with self._locks[plugin_id]:          # heavy models are not thread-safe; serialise per plugin
            out = p.run(clean)
        out.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 1))
        return out


def coerce(fields: list[Field], values: dict[str, Any]) -> dict[str, Any]:
    """Validate and type-cast incoming form values against the plugin's field spec."""
    out: dict[str, Any] = {}
    for f in fields:
        v = values.get(f.name, f.default)
        if v in (None, "") and f.required and f.type != "checkbox":
            raise ValueError(f"'{f.label}' is required")
        if f.type == "number" and v not in (None, ""):
            v = float(v)
            if f.min is not None and v < f.min:
                raise ValueError(f"'{f.label}' must be >= {f.min}")
            if f.max is not None and v > f.max:
                raise ValueError(f"'{f.label}' must be <= {f.max}")
            if f.step == 1:
                v = int(v)
        elif f.type == "select" and f.options and v not in f.options:
            raise ValueError(f"'{f.label}' must be one of {f.options}")
        elif f.type == "checkbox":
            v = bool(v) and str(v).lower() not in ("false", "0", "")
        out[f.name] = v
    return out


def parse_numbers(text: str) -> list[float]:
    import re
    vals = [float(t) for t in re.split(r"[,\s;]+", text.strip()) if t]
    if not vals:
        raise ValueError("no numbers found")
    return vals
