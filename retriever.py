# #added
# import os
# import json
# import math
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from typing import List, Dict, Tuple

# import config as C

# # Optional deps
# try:
#     from rank_bm25 import BM25Okapi
# except Exception:
#     BM25Okapi = None

# try:
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity
# except Exception:
#     TfidfVectorizer = None
#     cosine_similarity = None

# # Optional OpenAI
# OPENAI_OK = False
# try:
#     import openai
#     from openai import OpenAI
#     if C.LLM_PROVIDER == "openai" and (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower())):
#         client = OpenAI()
#         OPENAI_OK = True
# except Exception:
#     OPENAI_OK = False


# def _norm(s: str) -> str:
#     import re
#     s = re.sub(r"\s+", " ", str(s or "")).strip()
#     return s


# def build_corpus(df: pd.DataFrame) -> List[str]:
#     return (
#         df["problem_text"].fillna("") + " " +
#         df["solution_summary"].fillna("") + " " +
#         df.get("tags", "").fillna("")
#     ).map(_norm).tolist()


# # ---------------- TF-IDF ----------------
# class TFIDFSearch:
#     def __init__(self, corpus_texts: List[str]):
#         if TfidfVectorizer is None:
#             self.vec = None
#             self.X = None
#             return
#         self.vec = TfidfVectorizer(lowercase=True, stop_words="english")
#         self.X = self.vec.fit_transform(corpus_texts)

#     def scores(self, query: str) -> np.ndarray:
#         if self.vec is None:
#             return np.zeros(0)
#         q = self.vec.transform([query])
#         sims = cosine_similarity(q, self.X).ravel()
#         return sims


# # ---------------- BM25 ----------------
# class BM25Search:
#     def __init__(self, corpus_texts: List[str]):
#         self.tokenized = [t.lower().split() for t in corpus_texts]
#         self.ok = BM25Okapi is not None and len(self.tokenized) > 0
#         self.bm25 = BM25Okapi(self.tokenized) if self.ok else None

#     def scores(self, query: str) -> np.ndarray:
#         if not self.ok:
#             return np.zeros(len(self.tokenized))
#         q = query.lower().split()
#         s = np.array(self.bm25.get_scores(q), dtype=float)
#         # Normalize to 0–1 for hybrid mixing
#         if s.max() > 0:
#             s = (s - s.min()) / (s.max() - s.min() + 1e-9)
#         return s


# # ---------------- Embeddings ----------------
# def _embed_texts(texts: List[str]) -> np.ndarray:
#     """Return (n, d) np array. If API not available, return zeros."""
#     if not OPENAI_OK or not C.USE_EMBEDDINGS:
#         return np.zeros((len(texts), 1), dtype=float)

#     # batch embed to be efficient
#     out_vecs = []
#     step = C.EMBED_BATCH
#     for i in range(0, len(texts), step):
#         chunk = texts[i:i+step]
#         resp = client.embeddings.create(model=C.EMBED_MODEL, input=chunk)
#         out_vecs.extend([np.array(e.embedding, dtype=float) for e in resp.data])
#     return np.vstack(out_vecs)


# def _embed_query(query: str, dim: int) -> np.ndarray:
#     if not OPENAI_OK or not C.USE_EMBEDDINGS:
#         return np.zeros((dim,), dtype=float)
#     resp = client.embeddings.create(model=C.EMBED_MODEL, input=[query])
#     return np.array(resp.data[0].embedding, dtype=float)


# def _cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
#     a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
#     b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
#     return a @ b.T  # (n, d) @ (d, ) -> (n,)


# class EmbedSearch:
#     def __init__(self, texts: List[str], cache_dir: Path, cache_file: str):
#         self.texts = texts
#         self.cache_path = cache_dir / cache_file
#         self.emb = None
#         self.dim = 0

#         cache_dir.mkdir(exist_ok=True, parents=True)

#         if self.cache_path.exists():
#             try:
#                 df = pd.read_parquet(self.cache_path)
#                 if len(df) == len(texts):
#                     self.emb = np.stack(df["emb"].to_list()).astype(float)
#             except Exception:
#                 self.emb = None

#         if self.emb is None:
#             self.emb = _embed_texts(texts)
#             self.dim = self.emb.shape[1] if self.emb.size else 1
#             # save cache
#             try:
#                 df = pd.DataFrame({"emb": [row.tolist() for row in self.emb]})
#                 df.to_parquet(self.cache_path, index=False)
#             except Exception:
#                 pass
#         else:
#             self.dim = self.emb.shape[1] if self.emb.size else 1

#     def scores(self, query: str) -> np.ndarray:
#         if self.emb is None or self.emb.size == 0:
#             return np.zeros(len(self.texts))
#         q = _embed_query(query, self.dim)
#         sims = _cos_sim(self.emb, q)
#         # Normalize 0–1 for mixing
#         sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-9)
#         return sims


# # ---------------- Hybrid retrieval ----------------
# class HybridRetriever:
#     def __init__(self, kb_df: pd.DataFrame):
#         self.df = kb_df.reset_index(drop=True).copy()
#         self.corpus = build_corpus(self.df)
#         self.tfidf = TFIDFSearch(self.corpus)
#         self.bm25  = BM25Search(self.corpus) if C.USE_BM25 else None
#         self.emb   = EmbedSearch(self.corpus, Path(C.CACHE_DIR), C.EMBED_CACHE_FILE) if C.USE_EMBEDDINGS else None

#     def search(self, query: str, top_k: int = 8) -> List[Dict]:
#         n = len(self.df)
#         if n == 0:
#             return []

#         s_tfidf = self.tfidf.scores(query) if self.tfidf else np.zeros(n)
#         s_bm25  = self.bm25.scores(query)  if self.bm25  else np.zeros(n)
#         s_emb   = self.emb.scores(query)   if self.emb   else np.zeros(n)

#         mix = C.WEIGHT_TFIDF * s_tfidf + C.WEIGHT_BM25 * s_bm25 + C.WEIGHT_EMB * s_emb
#         order = np.argsort(mix)[::-1][:max(top_k, C.RERANK_TOP_K)]
#         cand = self.df.iloc[order].copy()
#         cand["_score_mix"] = mix[order]

#         # Optional semantic re-rank on top bucket
#         if C.USE_EMBEDDINGS and self.emb is not None and OPENAI_OK:
#             # compute fresh embedding similarity to re-rank
#             sims = s_emb[order]
#             cand["_score_semantic"] = sims
#             cand = cand.sort_values(by=["_score_semantic", "_score_mix"], ascending=False)

#         return cand.head(top_k).to_dict(orient="records")


# def format_steps(steps_json: str) -> List[str]:
#     try:
#         return list(json.loads(steps_json))
#     except Exception:
#         return []






# retriever.py
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


# ---------- helpers ----------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def format_steps(steps_json: str | list | None) -> List[str]:
    if steps_json is None:
        return []
    if isinstance(steps_json, list):
        return [str(x) for x in steps_json]
    try:
        data = json.loads(steps_json)
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return []


# ---------- model / artifacts ----------
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class _IndexArtifacts:
    model_name: str
    nn: NearestNeighbors
    embeddings: np.ndarray
    kb_rows: pd.DataFrame
    search_text: List[str]


class HybridRetriever:
    """
    Embedding-based retriever with a clean API:
      - train(df)
      - save(path)
      - load(path)
      - search(query, top_k=5) -> List[dict]
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, model_name: str = DEFAULT_ST_MODEL):
        self.model_name = model_name
        self._artifacts: Optional[_IndexArtifacts] = None
        self._df: Optional[pd.DataFrame] = None
        self._model: Optional[SentenceTransformer] = None  # lazy cache
        if df is not None:
            self._df = df.copy()

    # ---------- training ----------
    def train(self, df: Optional[pd.DataFrame] = None) -> None:
        if df is not None:
            self._df = df.copy()
        if self._df is None:
            raise ValueError("HybridRetriever.train: no DataFrame available")

        kb = self._df.copy()
        if "search_text" not in kb.columns:
            kb["search_text"] = (
                kb["problem_text"].fillna("") + " " +
                kb["solution_summary"].fillna("") + " " +
                kb.get("tags", "").fillna("")
            ).map(_norm)

        search_text = kb["search_text"].tolist()

        # load / cache model
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        embeddings = self._model.encode(
            search_text,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine-friendly
        )

        nn = NearestNeighbors(
            n_neighbors=min(10, max(1, len(search_text))),
            metric="cosine",
            algorithm="auto",
        )
        nn.fit(embeddings)

        keep_cols = [
            "issue_id", "problem_text", "solution_summary", "tags",
            "confidence", "steps_solution", "fallback", "evidence_comment_ids",
            "search_text",
        ]
        kb_rows = kb[[c for c in keep_cols if c in kb.columns]].copy()

        self._artifacts = _IndexArtifacts(
            model_name=self.model_name,
            nn=nn,
            embeddings=embeddings,
            kb_rows=kb_rows,
            search_text=search_text,
        )

    # ---------- persistence ----------
    def save(self, path: str | Path) -> None:
        if self._artifacts is None:
            raise ValueError("HybridRetriever.save: no trained artifacts to save")

        path = Path(path)
        out_dir = path if path.is_dir() else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._artifacts.nn, out_dir / "index.joblib")
        np.save(out_dir / "embeddings.npy", self._artifacts.embeddings)
        self._artifacts.kb_rows.to_parquet(out_dir / "kb_embed.parquet", index=False)

        manifest = {
            "model_name": self._artifacts.model_name,
            "num_rows": int(self._artifacts.embeddings.shape[0]),
            "dim": int(self._artifacts.embeddings.shape[1]) if self._artifacts.embeddings.size else 0,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        in_dir = path if path.is_dir() else path.parent

        nn = joblib.load(in_dir / "index.joblib")
        embeddings = np.load(in_dir / "embeddings.npy")
        kb_rows = pd.read_parquet(in_dir / "kb_embed.parquet")
        try:
            manifest = json.loads((in_dir / "manifest.json").read_text(encoding="utf-8"))
            model_name = manifest.get("model_name", DEFAULT_ST_MODEL)
        except Exception:
            model_name = DEFAULT_ST_MODEL

        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name)  # ensure model present

        search_text = kb_rows.get("search_text", pd.Series([""] * len(kb_rows))).astype(str).tolist()

        self._artifacts = _IndexArtifacts(
            model_name=model_name,
            nn=nn,
            embeddings=embeddings,
            kb_rows=kb_rows,
            search_text=search_text,
        )

    # ---------- search ----------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self._artifacts is None:
            if self._df is None:
                raise ValueError("HybridRetriever.search: index not loaded/trained")
            self.train(self._df)

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        art = self._artifacts
        assert art is not None

        qv = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = art.nn.kneighbors(qv, n_neighbors=min(top_k, len(art.kb_rows)))
        distances = distances[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for d, i in zip(distances, indices):
            row = art.kb_rows.iloc[int(i)].to_dict()
            row["_score_semantic"] = float(np.clip(1.0 - float(d), 0.0, 1.0))
            results.append(row)

        results.sort(key=lambda r: r.get("_score_semantic", 0.0), reverse=True)
        return results
