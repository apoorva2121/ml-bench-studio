# ML Bench Studio — one console to try every model in this workspace

A small FastAPI app with a plain-HTML front end. The home screen lists every model as a tile; open one,
load an example or type your own input, press Run, and the result comes back with the numbers, charts,
images or citations that model produces. Every model is a **plugin**: one Python file that declares its
inputs, its examples and a `run()` function. The UI is generated from that declaration, so adding the
next model never touches the front end.

## Run it

```bash
cd studio
pip install fastapi uvicorn pillow            # everything else is already installed by the model repos
KMP_DUPLICATE_LIB_OK=TRUE uvicorn app:app --port 8010
# open http://127.0.0.1:8010
```

Models whose artifacts have not been trained yet show an "artifacts missing" pill and explain what to run.

## Models wired in

| Plugin | What you can do | Backed by |
|---|---|---|
| Ames House Prices | edit 10 key features of a real house, re-price it with the blend | `house-prices/artifacts/model.joblib` |
| M4 Forecasting Baselines | paste any series; Naive2 / Theta / ETS / seasonal naive + ensemble, scored on a held-out tail | `m4-forecasting` |
| Bayesian Marketing Mix | adstock carry-over and Hill saturation per channel, from the fitted posterior or manual params | `bayesian-mmm/reports` |
| Serving Template Model | the validated 7-feature request the FastAPI service accepts | `ml-serving-template/artifacts` |
| Scientific Retrieval (BEIR) | BM25 over SciFact / NFCorpus with gold labels for benchmark queries | `retrieval-eval/data` |
| MedMNIST Image Classification | classify a test image with either CNN, see Grad-CAM | `medmnist-classification/artifacts` |
| Medical RAG with Guardrails | ask a health question, watch each guardrail stage decide; optional LLM key held in memory only | `medical-rag/artifacts/index` |

## Adding a model (plug and play)

Create `plugins/<name>.py`:

```python
from registry import Plugin, Field, Example, ML_BENCH

def run(v: dict) -> dict:
    ...                                   # load lazily, cache with functools.lru_cache
    return {"summary": "...", "metrics": [{"label": "AUC", "value": "0.93"}],
            "series": {"x": [...], "series": [{"name": "pred", "y": [...]}]}}

PLUGIN = Plugin(
    id="my-model", title="My Model", tagline="one line", category="Tabular", repo="my-model",
    headline_metric="the number from the README", icon="cube",
    fields=[Field("x", "Input x", "number", 1.0, min=0, max=10)],
    examples=[Example("Default", {"x": 1.0})],
    run=run,
)
```

Restart the server. The tile, the form, the example chips, validation and rendering all come from
the contract. Result keys the renderer understands: `summary`, `metrics`, `table`, `series`, `bars`,
`image` (+ `image_plain`, `image_caption`), `text`, `citations`, `details`, `warnings`.

Rules the registry enforces: field types and ranges are validated server-side before `run()`;
each plugin runs under its own lock (torch and sentence-transformers are not thread-safe); a plugin
that fails to import is listed on the home page instead of crashing the app; password fields are
never echoed back and the UI clears them after each run.

## Design

Generated with the ui-ux-pro-max skill (`design-system/ml-bench-studio/MASTER.md`): "AI-Native UI"
style, Inter, purple primary / cyan accent, dense dashboard spacing, subtle motion, light and dark
themes. Accessibility: 44 px targets, visible focus rings, labelled fields with helper text, errors
next to the form, `prefers-reduced-motion` respected, SVG icons (no emoji), charts carry `role="img"`
labels and tooltips.

## Status (23 Sep 2026) — work in progress

All seven plugins pass their examples through the API (`/api/run/<id>`), and the home grid, model
screen, forms, examples, validation errors, metrics, images and tables render. **Known bug:** on the
M4 screen the tab's renderer freezes when the line chart is inserted, even though the chart function
itself returns in ~50 ms in Node. Suspected cause is SVG sizing inside the CSS grid; wrapping the SVG in a
fixed-aspect container did not resolve it. Next step: render charts to `<canvas>` or a chart library
instead of inline SVG. UI work is paused until the models are finished.

## Licence

[MIT](LICENSE).
