# chat_cli.py
import os, json, re, sys, pandas as pd
from pathlib import Path
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to use OpenAI for nicer phrasing if configured
USE_OPENAI = False
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
    try:
        from openai import OpenAI
        client = OpenAI()
        USE_OPENAI = True
    except Exception:
        USE_OPENAI = False

KB_PATH = Path("out/kb_heuristic.csv")
CONF_RANK = {"high": 3, "medium": 2, "low": 1}

def _conf_rank(s):
    return CONF_RANK.get(str(s or "").strip().lower(), 0)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def load_kb():
    if not KB_PATH.exists():
        raise SystemExit("KB not found at out/kb_heuristic.csv. Run: python main.py")
    df = pd.read_csv(KB_PATH)
    df["search_text"] = (
        df["problem_text"].fillna("") + " " +
        df["solution_summary"].fillna("") + " " +
        df.get("tags","").fillna("")
    ).map(norm)
    return df

def build_vectorizer(texts):
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(texts)
    return vec, X

def search(df, vec, X, query, k=5):
    q = vec.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top_idx = sims.argsort()[::-1][:k]
    rows = []
    for i in top_idx:
        row = df.iloc[i].to_dict()
        row["_score"] = float(sims[i])
        rows.append(row)
    return rows

def pick_best(hits):
    return max(hits, key=lambda h: (_conf_rank(h.get("confidence")), float(h.get("_score", 0.0))))

def format_answer_from_hit(hit):
    # Build a friendly, self-contained answer without OpenAI
    steps = []
    try:
        steps = json.loads(hit.get("steps_solution","[]"))
    except Exception:
        pass
    parts = []
    # One-line helpful summary
    summary = hit.get("solution_summary","").strip()
    if summary:
        parts.append(summary)
    # Steps
    if steps:
        parts.append("Try this:")
        for i, s in enumerate(steps, 1):
            parts.append(f"{i}. {s}")
    # Fallback
    if hit.get("fallback"):
        parts.append(f"If that doesn’t work: {hit['fallback']}")
    # Confidence/tag hint
    conf = hit.get("confidence","")
    tags_txt = ""
    try:
        tags = json.loads(hit.get("tags","[]"))
        if tags:
            tags_txt = f" | Tags: {', '.join(tags)}"
    except Exception:
        pass
    # Footer with small meta (optional)
    parts.append(f"(Answer confidence: {conf}{tags_txt}; last verified {date.today().isoformat()})")
    return "\n".join(parts)

def rewrite_with_openai(user_q, raw_answer, hit):
    try:
        prompt = f"""You are a concise, friendly support bot. The user asked:
"{user_q}"

You have a knowledge-base hit (confidence={hit.get('confidence','')}), and a draft answer:

{raw_answer}

Please rewrite the answer to be:
- one short intro line addressing the user's problem,
- then a numbered list of up to 4 steps,
- then one single-line fallback starting with "If this still fails, ...".
Keep it crisp and helpful. Do not invent steps."""
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role":"system","content":"You are a helpful support assistant."},
                {"role":"user","content": prompt}
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return raw_answer

def answer(df, vec, X, query):
    hits = search(df, vec, X, query, k=5)
    best = pick_best(hits)
    raw = format_answer_from_hit(best)
    if USE_OPENAI:
        return rewrite_with_openai(query, raw, best)
    return raw

def main():
    df = load_kb()
    vec, X = build_vectorizer(df["search_text"].tolist())

    # One-shot from CLI args
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(answer(df, vec, X, q))
        return

    # Interactive chat
    print("AI Jenerator Chat — type your question. Enter to quit.")
    while True:
        q = input("\nYou: ").strip()
        if not q:
            break
        print("\nAI:", answer(df, vec, X, q))

if __name__ == "__main__":
    main()
