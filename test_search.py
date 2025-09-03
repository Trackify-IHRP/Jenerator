# # test_search.py
# import os, sys, json, re, pickle
# from pathlib import Path
# import argparse
# import pandas as pd

# import config as C
# from retriever import HybridRetriever, format_steps

# KB_PATH = Path("out/kb_heuristic.csv")
# OUT_DIR = Path("out")

# CONF_RANK = {"high": 3, "medium": 2, "low": 1}
# def _conf_rank(s):
#     return CONF_RANK.get(str(s or "").strip().lower(), 0)

# def norm(s: str) -> str:
#     s = re.sub(r"\s+", " ", str(s or "")).strip()
#     return s

# def load_kb():
#     if not KB_PATH.exists():
#         raise SystemExit(f"KB not found at {KB_PATH}. Run: python main.py")
#     df = pd.read_csv(KB_PATH)
#     # Precompute “search_text” (also used by TF-IDF fallback training)
#     df["search_text"] = (
#         df["problem_text"].fillna("") + " " +
#         df["solution_summary"].fillna("") + " " +
#         df.get("tags", "").fillna("")
#     ).map(norm)
#     return df

# def score_with_conf(conf: str, base: float) -> float:
#     return base + 0.05 * _conf_rank(conf)

# def pick_best(hits):
#     return max(
#         hits,
#         key=lambda h: score_with_conf(
#             h.get("confidence", ""),
#             float(h.get("_score_semantic", h.get("_score_mix", 0.0)))
#         )
#     )

# def pretty_print(hit):
#     score = float(hit.get("_score_semantic", hit.get("_score_mix", 0.0)))
#     print(f"\nScore: {score:.3f}  |  Issue: {hit.get('issue_id','')}  |  Confidence: {hit.get('confidence','')}")
#     print(f"Problem: {hit.get('problem_text','')}")
#     print(f"Summary: {hit.get('solution_summary','')}")
#     steps = format_steps(hit.get("steps_solution","[]"))
#     if steps:
#         print("Steps:")
#         for s in steps:
#             print(f"  • {s}")
#     if hit.get("fallback"):
#         print(f"Fallback: {hit['fallback']}")
#     try:
#         tags = json.loads(hit.get("tags","[]"))
#         if tags:
#             print("Tags:", ", ".join(tags))
#     except Exception:
#         pass
#     print(f"Evidence comment IDs: {hit.get('evidence_comment_ids','[]')}")
#     print("-"*80)

# def maybe_generate_final_answer(query: str, hit: dict) -> str:
#     if not C.ANSWER_WITH_LLM or C.LLM_PROVIDER != "openai" or not os.getenv("OPENAI_API_KEY"):
#         steps = format_steps(hit.get("steps_solution","[]"))
#         bullets = "\n".join([f"- {s}" for s in steps]) if steps else ""
#         tags = ", ".join(json.loads(hit.get("tags","[]") or "[]")) if hit.get("tags") else ""
#         reply = (
# f"""**Issue:** {hit.get('problem_text','')}

# **Likely fix/workaround:** {hit.get('solution_summary','')}

# **Try these steps:**
# {bullets if bullets else '- Retry the action.\n- Restart the app/browser.\n- If it persists, open a support ticket with a screenshot and error text.'}

# **Fallback:** {hit.get('fallback','Open a support case with logs and screenshots.')}

# **Confidence:** {hit.get('confidence','')}{('  |  Tags: ' + tags) if tags else ''}"""
#         )
#         return reply

#     from openai import OpenAI
#     client = OpenAI()
#     steps = format_steps(hit.get("steps_solution","[]"))
#     sys_prompt = (
#         "You are a helpful support assistant. Keep answers concise, actionable, and friendly. "
#         "Prefer bullet points. Never invent steps not implied by the KB."
#     )
#     user_prompt = json.dumps({
#         "user_query": query,
#         "kb_record": {
#             "issue_id": hit.get("issue_id",""),
#             "problem_text": hit.get("problem_text",""),
#             "solution_summary": hit.get("solution_summary",""),
#             "steps": steps,
#             "fallback": hit.get("fallback",""),
#             "confidence": hit.get("confidence",""),
#             "tags": hit.get("tags","[]"),
#         }
#     }, ensure_ascii=False)

#     resp = client.chat.completions.create(
#         model=C.OPENAI_MODEL,
#         temperature=0.2,
#         messages=[
#             {"role": "system", "content": sys_prompt},
#             {"role": "user", "content": user_prompt},
#         ],
#     )
#     return resp.choices[0].message.content.strip()

# # ---------------------------
# # Training implementations
# # ---------------------------

# def train_with_hybrid(df):
#     """
#     If your HybridRetriever supports training/persisting, use it.
#     """
#     retriever = HybridRetriever(df)
#     trained = False

#     # Try common method names defensively
#     if hasattr(retriever, "train"):
#         retriever.train(df)  # type: ignore
#         trained = True

#     # Try to persist an index if possible
#     if hasattr(retriever, "save"):
#         OUT_DIR.mkdir(parents=True, exist_ok=True)
#         retriever.save(str(OUT_DIR / "hybrid.index"))  # type: ignore

#     return trained

# def train_with_tfidf(df):
#     """
#     Fallback training: builds a TF-IDF index and saves artifacts to out/.
#     """
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     vec = TfidfVectorizer(lowercase=True, stop_words="english")
#     X = vec.fit_transform(df["search_text"].tolist())

#     OUT_DIR.mkdir(parents=True, exist_ok=True)

#     with open(OUT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
#         pickle.dump(vec, f)

#     with open(OUT_DIR / "tfidf_matrix.pkl", "wb") as f:
#         pickle.dump(X, f)  # pickling the scipy sparse matrix is fine

#     # Minimal metadata for downstream usage
#     keep_cols = [
#         "issue_id","problem_text","solution_summary","tags",
#         "confidence","steps_solution","fallback","evidence_comment_ids"
#     ]
#     meta = df[[c for c in keep_cols if c in df.columns]].copy()
#     meta.to_json(OUT_DIR / "kb_meta.json", orient="records", force_ascii=False, indent=2)

# def cmd_train():
#     df = load_kb()

#     print("→ Training from KB:", KB_PATH)
#     # Prefer HybridRetriever native training if available
#     used_hybrid = train_with_hybrid(df)
#     if used_hybrid:
#         print("✓ Hybrid retriever training complete.")
#         print(f"✓ (If supported) Index saved to: {OUT_DIR / 'hybrid.index'}")
#     else:
#         print("↪ Hybrid retriever has no train/save methods; falling back to TF-IDF.")
#         train_with_tfidf(df)
#         print("✓ TF-IDF artifacts saved to:")
#         print(f"   - {OUT_DIR / 'tfidf_vectorizer.pkl'}")
#         print(f"   - {OUT_DIR / 'tfidf_matrix.pkl'}")
#         print(f"   - {OUT_DIR / 'kb_meta.json'}")

# def cmd_query(args):
#     df = load_kb()
#     retriever = HybridRetriever(df)

#     # If your HybridRetriever can load a saved index, try it
#     idx_path = OUT_DIR / "hybrid.index"
#     if hasattr(retriever, "load") and idx_path.exists():
#         try:
#             retriever.load(str(idx_path))  # type: ignore
#         except Exception:
#             pass  # non-fatal; it can still search from df

#     query = " ".join(args.query).strip()
#     if not query:
#         print("No query provided. Example:")
#         print("  python test_search.py query \"Power Apps ID not saving to Excel\"")
#         sys.exit(2)

#     hits = retriever.search(query, top_k=5)

#     if args.top3:
#         for h in hits[:3]:
#             pretty_print(h)
#         return

#     if args.json:
#         best = pick_best(hits)
#         print(json.dumps(best, ensure_ascii=False, indent=2))
#         return

#     best = pick_best(hits)
#     pretty_print(best)
#     print("\n--- Chatbot Answer ---")
#     print(maybe_generate_final_answer(query, best))

# def build_arg_parser():
#     p = argparse.ArgumentParser(
#         description="KB trainer & search CLI. Default action is 'train'."
#     )
#     sub = p.add_subparsers(dest="cmd")

#     # train (default)
#     sub.add_parser("train", help="Build and cache the index from the KB")

#     # query
#     q = sub.add_parser("query", help="Run a search over the KB")
#     q.add_argument("--top3", action="store_true", help="Show top 3 matches (debug)")
#     q.add_argument("--json", action="store_true", help="Return best match as JSON")
#     q.add_argument("query", nargs="*", help="Query text")

#     return p

# def main():
#     parser = build_arg_parser()
#     args = parser.parse_args()

#     # Default to "train" when no subcommand is provided
#     cmd = args.cmd or "train"

#     if cmd == "train":
#         cmd_train()
#     elif cmd == "query":
#         cmd_query(args)
#     else:
#         parser.print_help()
#         sys.exit(2)

# if __name__ == "__main__":
#     main()




# test_search.py
import os, sys, json, re, pickle, time, traceback
from pathlib import Path
import argparse
import pandas as pd
from datetime import datetime

import config as C
from retriever import HybridRetriever, format_steps

KB_PATH = Path("out/kb_heuristic.csv")
OUT_DIR = Path("out")

CONF_RANK = {"high": 3, "medium": 2, "low": 1}
def _conf_rank(s):
    return CONF_RANK.get(str(s or "").strip().lower(), 0)

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s

def load_kb():
    if not KB_PATH.exists():
        raise SystemExit(f"KB not found at {KB_PATH}. Run: python main.py")
    df = pd.read_csv(KB_PATH)
    # Precompute “search_text” (also used by TF-IDF fallback training)
    df["search_text"] = (
        df["problem_text"].fillna("") + " " +
        df["solution_summary"].fillna("") + " " +
        df.get("tags", "").fillna("")
    ).map(norm)
    return df

def score_with_conf(conf: str, base: float) -> float:
    return base + 0.05 * _conf_rank(conf)

def pick_best(hits):
    return max(
        hits,
        key=lambda h: score_with_conf(
            h.get("confidence", ""),
            float(h.get("_score_semantic", h.get("_score_mix", 0.0)))
        )
    )

def pretty_print(hit):
    score = float(hit.get("_score_semantic", hit.get("_score_mix", 0.0)))
    print(f"\nScore: {score:.3f}  |  Issue: {hit.get('issue_id','')}  |  Confidence: {hit.get('confidence','')}")
    print(f"Problem: {hit.get('problem_text','')}")
    print(f"Summary: {hit.get('solution_summary','')}")
    steps = format_steps(hit.get("steps_solution","[]"))
    if steps:
        print("Steps:")
        for s in steps:
            print(f"  • {s}")
    if hit.get("fallback"):
        print(f"Fallback: {hit['fallback']}")
    try:
        tags = json.loads(hit.get("tags","[]"))
        if tags:
            print("Tags:", ", ".join(tags))
    except Exception:
        pass
    print(f"Evidence comment IDs: {hit.get('evidence_comment_ids','[]')}")
    print("-"*80)

def maybe_generate_final_answer(query: str, hit: dict) -> str:
    if not C.ANSWER_WITH_LLM or C.LLM_PROVIDER != "openai" or not os.getenv("OPENAI_API_KEY"):
        steps = format_steps(hit.get("steps_solution","[]"))
        bullets = "\n".join([f"- {s}" for s in steps]) if steps else ""
        tags = ", ".join(json.loads(hit.get("tags","[]") or "[]")) if hit.get("tags") else ""
        reply = (
f"""**Issue:** {hit.get('problem_text','')}

**Likely fix/workaround:** {hit.get('solution_summary','')}

**Try these steps:**
{bullets if bullets else '- Retry the action.\n- Restart the app/browser.\n- If it persists, open a support ticket with a screenshot and error text.'}

**Fallback:** {hit.get('fallback','Open a support case with logs and screenshots.')}

**Confidence:** {hit.get('confidence','')}{('  |  Tags: ' + tags) if tags else ''}"""
        )
        return reply

    from openai import OpenAI
    client = OpenAI()
    steps = format_steps(hit.get("steps_solution","[]"))
    sys_prompt = (
        "You are a helpful support assistant. Keep answers concise, actionable, and friendly. "
        "Prefer bullet points. Never invent steps not implied by the KB."
    )
    user_prompt = json.dumps({
        "user_query": query,
        "kb_record": {
            "issue_id": hit.get("issue_id",""),
            "problem_text": hit.get("problem_text",""),
            "solution_summary": hit.get("solution_summary",""),
            "steps": steps,
            "fallback": hit.get("fallback",""),
            "confidence": hit.get("confidence",""),
            "tags": hit.get("tags","[]"),
        }
    }, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=C.OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Training implementations
# ---------------------------

def train_with_hybrid(df):
    """
    If your HybridRetriever supports training/persisting, use it.
    """
    retriever = HybridRetriever(df)
    trained = False

    # Try common method names defensively
    if hasattr(retriever, "train"):
        retriever.train(df)  # type: ignore
        trained = True

    # Try to persist an index if possible
    if hasattr(retriever, "save"):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        retriever.save(str(OUT_DIR / "hybrid.index"))  # type: ignore

    return trained

def train_with_tfidf(df):
    """
    Fallback training: builds a TF-IDF index and saves artifacts to out/.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(df["search_text"].tolist())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)

    with open(OUT_DIR / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(X, f)  # pickling the scipy sparse matrix is fine

    # Minimal metadata for downstream usage
    keep_cols = [
        "issue_id","problem_text","solution_summary","tags",
        "confidence","steps_solution","fallback","evidence_comment_ids"
    ]
    meta = df[[c for c in keep_cols if c in df.columns]].copy()
    meta.to_json(OUT_DIR / "kb_meta.json", orient="records", force_ascii=False, indent=2)

def do_train():
    print(f"[{ts()}] → Training from KB: {KB_PATH}")
    df = load_kb()

    used_hybrid = train_with_hybrid(df)
    if used_hybrid:
        print(f"[{ts()}] ✓ Hybrid retriever training complete.")
        print(f"[{ts()}] ✓ (If supported) Index saved to: {OUT_DIR / 'hybrid.index'}")
    else:
        print(f"[{ts()}] ↪ Hybrid retriever has no train/save methods; falling back to TF-IDF.")
        train_with_tfidf(df)
        print(f"[{ts()}] ✓ TF-IDF artifacts saved to:")
        print(f"[{ts()}]    - {OUT_DIR / 'tfidf_vectorizer.pkl'}")
        print(f"[{ts()}]    - {OUT_DIR / 'tfidf_matrix.pkl'}")
        print(f"[{ts()}]    - {OUT_DIR / 'kb_meta.json'}")

def cmd_train_once(_args=None):
    do_train()

def cmd_query(args):
    df = load_kb()
    retriever = HybridRetriever(df)

    # If your HybridRetriever can load a saved index, try it
    idx_path = OUT_DIR / "hybrid.index"
    if hasattr(retriever, "load") and idx_path.exists():
        try:
            retriever.load(str(idx_path))  # type: ignore
        except Exception:
            pass  # non-fatal; it can still search from df

    query = " ".join(args.query).strip()
    if not query:
        print("No query provided. Example:")
        print("  python test_search.py query \"Power Apps ID not saving to Excel\"")
        sys.exit(2)

    hits = retriever.search(query, top_k=5)

    if args.top3:
        for h in hits[:3]:
            pretty_print(h)
        return

    if args.json:
        best = pick_best(hits)
        print(json.dumps(best, ensure_ascii=False, indent=2))
        return

    best = pick_best(hits)
    pretty_print(best)
    print("\n--- Chatbot Answer ---")
    print(maybe_generate_final_answer(query, best))

def cmd_watch(args):
    """
    Watch mode: trains immediately, then retrains when KB changes.
    Stops on Ctrl+C.
    """
    interval = getattr(args, "interval", 5.0)  # robust default if not set
    last_mtime = None
    last_size = None

    def current_sig():
        try:
            stat = KB_PATH.stat()
            return (stat.st_mtime_ns, stat.st_size)
        except FileNotFoundError:
            return None

    print(f"[{ts()}] 👀 Watching {KB_PATH} (poll every {interval}s). Press Ctrl+C to stop.")
    # Initial train
    do_train()
    last = current_sig()
    if last:
        last_mtime, last_size = last

    try:
        while True:
            time.sleep(interval)
            sig = current_sig()
            if sig is None:
                # File missing — keep waiting
                print(f"[{ts()}] … waiting for KB file to appear")
                continue
            mtime, size = sig
            if last_mtime is None or mtime != last_mtime or size != last_size:
                # Debounce a bit to let OneDrive/SharePoint finish writing
                time.sleep(max(1.0, min(5.0, interval/2)))
                print(f"[{ts()}] 🔄 Change detected (mtime/size). Retraining...")
                try:
                    do_train()
                except Exception:
                    print(f"[{ts()}] ❌ Training error:\n{traceback.format_exc()}")
                last_mtime, last_size = mtime, size
            else:
                # Heartbeat log
                print(f"[{ts()}] … no change")
    except KeyboardInterrupt:
        print(f"\n[{ts()}] 📴 Stopped watching.")

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="KB trainer & search CLI. Default action is 'watch' (continuous training)."
    )
    sub = p.add_subparsers(dest="cmd")

    # watch (default)
    w = sub.add_parser("watch", help="Train now, then retrain when KB file changes")
    w.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds (default: 5)")

    # train once
    sub.add_parser("train", help="Build and cache the index from the KB once")

    # query
    q = sub.add_parser("query", help="Run a search over the KB")
    q.add_argument("--top3", action="store_true", help="Show top 3 matches (debug)")
    q.add_argument("--json", action="store_true", help="Return best match as JSON")
    q.add_argument("query", nargs="*", help="Query text")

    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Default to "watch" when no subcommand is provided
    cmd = args.cmd or "watch"

    if cmd == "watch":
        # Ensure interval exists even if invoked without subcommand
        if not hasattr(args, "interval"):
            args.interval = 5.0
        cmd_watch(args)
    elif cmd == "train":
        cmd_train_once(args)
    elif cmd == "query":
        cmd_query(args)
    else:
        parser.print_help()
        sys.exit(2)

if __name__ == "__main__":
    main()
