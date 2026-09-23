"""Retrieval eval — search SciFact / NFCorpus with the BM25 retriever from ml-bench/retrieval-eval."""

from __future__ import annotations

import sys
from functools import lru_cache

import pandas as pd

from registry import ML_BENCH, Example, Field, Plugin

REPO = ML_BENCH / "retrieval-eval"
sys.path.insert(0, str(REPO / "src"))
DATASETS = [d for d in ("scifact", "nfcorpus") if (REPO / "data" / f"{d}_corpus.parquet").exists()]


@lru_cache(maxsize=2)
def _index(ds: str):
    from retrieval_eval.retrievers import BM25Retriever
    corpus = pd.read_parquet(REPO / "data" / f"{ds}_corpus.parquet")
    texts = (corpus["title"].fillna("") + " " + corpus["text"].fillna("")).tolist()
    return BM25Retriever(corpus["_id"].astype(str).tolist(), texts, k1=0.9, b=0.4), corpus.set_index(corpus["_id"].astype(str))


def _gold(ds: str, query: str) -> set[str]:
    try:
        q = pd.read_parquet(REPO / "data" / f"{ds}_queries.parquet")
        qr = pd.read_parquet(REPO / "data" / f"{ds}_qrels_test.parquet")
        qid = q.loc[q["text"].str.strip().str.lower() == query.strip().lower(), "_id"]
        if qid.empty:
            return set()
        qcol = "query-id" if "query-id" in qr.columns else qr.columns[0]
        dcol = "corpus-id" if "corpus-id" in qr.columns else qr.columns[1]
        return set(qr.loc[qr[qcol].astype(str) == str(qid.iloc[0]), dcol].astype(str))
    except Exception:
        return set()


def run(v: dict) -> dict:
    ds, k = v["dataset"], int(v["k"])
    bm25, corpus = _index(ds)
    ranked = bm25.rank(v["query"], k)
    gold = _gold(ds, v["query"])
    rows = []
    for r, (did, score) in enumerate(ranked, 1):
        d = corpus.loc[did]
        rows.append({"rank": r, "score": round(score, 2), "relevant": "yes" if did in gold else ("" if not gold else "no"),
                     "title": (d["title"] or "")[:90], "snippet": (d["text"] or "")[:180]})
    hits = sum(1 for did, _ in ranked if did in gold)
    metrics = [{"label": "Corpus size", "value": f"{len(corpus):,}"}, {"label": "Top-k", "value": str(k)}]
    if gold:
        metrics += [{"label": f"Relevant in top-{k}", "value": f"{hits}/{len(gold)}", "hint": "this query is in the BEIR test set"}]
    return {"summary": f"BM25 over {ds}: {len(ranked)} results" + (f", {hits} relevant" if gold else ""),
            "metrics": metrics, "table": {"columns": ["rank", "score", "relevant", "title", "snippet"], "rows": rows},
            "warnings": [] if gold else ["Not a benchmark query — no relevance labels; judge the ranking yourself."]}


def _examples() -> list[Example]:
    ex = []
    for ds in DATASETS:
        try:
            q = pd.read_parquet(REPO / "data" / f"{ds}_queries.parquet")
            for t in q["text"].head(2):
                ex.append(Example(f"{ds}: {t[:50]}…" if len(t) > 50 else f"{ds}: {t}", {"dataset": ds, "query": t, "k": 10}))
        except Exception:
            pass
    ex.append(Example("Free text (no labels)", {"dataset": DATASETS[0] if DATASETS else "scifact", "query": "vitamin D supplementation and bone density", "k": 10}))
    return ex


PLUGIN = Plugin(
    id="retrieval-eval", title="Scientific Retrieval (BEIR)", tagline="BM25 over SciFact and NFCorpus, with relevance labels",
    category="Retrieval", repo="retrieval-eval", headline_metric="Hybrid RRF nDCG@10 0.687 SciFact · 0.336 NFCorpus", icon="search",
    description="Run a query against a BEIR corpus with the same BM25 configuration as the benchmark. Test-set queries are "
                "scored against their gold labels; free text just shows the ranking. (Dense/hybrid needs the corpus embedded first — kept out of the studio to stay fast.)",
    fields=[Field("dataset", "Dataset", "select", DATASETS[0] if DATASETS else "scifact", options=DATASETS or ["scifact"]),
            Field("query", "Query", "textarea", "vitamin D supplementation and bone density"),
            Field("k", "Results", "number", 10, min=1, max=50, step=1)],
    examples=_examples(), run=run, heavy=True,
    status="ready" if DATASETS else "needs-artifacts",
)
