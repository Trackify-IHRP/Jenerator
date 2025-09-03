# app_streamlit.py
import os, json, re, pandas as pd
from pathlib import Path
from datetime import date
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI rewrite
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

@st.cache_data
def load_kb():
    if not KB_PATH.exists():
        raise RuntimeError("KB not found at out/kb_heuristic.csv. Run: python main.py")
    df = pd.read_csv(KB_PATH)
    df["search_text"] = (
        df["problem_text"].fillna("") + " " +
        df["solution_summary"].fillna("") + " " +
        df.get("tags","").fillna("")
    ).map(lambda s: re.sub(r"\s+"," ",str(s)).strip())
    return df

@st.cache_resource
def build_vectorizer(texts):
    vec = TfidfVectorizer(lowercase=True, stop_words="english")
    X = vec.fit_transform(texts)
    return vec, X

def _conf_rank(s):
    return CONF_RANK.get(str(s or "").strip().lower(), 0)

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
    steps = []
    try:
        steps = json.loads(hit.get("steps_solution","[]"))
    except Exception:
        pass
    parts = []
    summary = (hit.get("solution_summary","") or "").strip()
    if summary:
        parts.append(summary)
    if steps:
        parts.append("Try this:")
        for i, s in enumerate(steps, 1):
            parts.append(f"{i}. {s}")
    if hit.get("fallback"):
        parts.append(f"If that doesn’t work: {hit['fallback']}")
    conf = hit.get("confidence","")
    tags_txt = ""
    try:
        tags = json.loads(hit.get("tags","[]"))
        if tags:
            tags_txt = f" | Tags: {', '.join(tags)}"
    except Exception:
        pass
    parts.append(f"(Answer confidence: {conf}{tags_txt}; last verified {date.today().isoformat()})")
    return "\n".join(parts)

def rewrite_with_openai(user_q, raw_answer, hit):
    try:
        prompt = f"""You are a concise, friendly support bot. The user asked:
"{user_q}"

Here is a draft answer:
{raw_answer}

Rewrite it as a short intro + numbered steps (max 4) + single-line fallback starting with "If this still fails, ...".
Do not invent steps."""
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

st.set_page_config(page_title="AI Jenerator Chat", page_icon="💬", layout="centered")
st.title("AI Jenerator Chat 💬")

df = load_kb()
vec, X = build_vectorizer(df["search_text"].tolist())

if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.chat_input("Type your question…")
if user_q:
    hits = search(df, vec, X, user_q, k=5)
    best = pick_best(hits)
    answer = format_answer_from_hit(best)
    if USE_OPENAI:
        answer = rewrite_with_openai(user_q, answer, best)
    st.session_state.history.append(("user", user_q))
    st.session_state.history.append(("ai", answer))

for role, msg in st.session_state.history:
    with st.chat_message("user" if role=="user" else "assistant"):
        st.markdown(msg)
