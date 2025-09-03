# import os, json, textwrap, pathlib, re
# from datetime import date
# from dotenv import load_dotenv
# import pandas as pd
# from tqdm import tqdm

# from tenacity import retry, stop_after_attempt, wait_exponential
# import requests

# import config as C

# OUT_DIR = pathlib.Path("out"); OUT_DIR.mkdir(exist_ok=True)

# def load_tables():
#     issues = pd.read_excel(C.EXCEL_PATH, sheet_name=C.ISSUES_SHEET)
#     comments = pd.read_excel(C.EXCEL_PATH, sheet_name=C.COMMENTS_SHEET)
#     # Standardize column names
#     issues = issues.rename(columns={
#         C.ISSUE_COLS["issue_id"]: "issue_id",
#         C.ISSUE_COLS["problem_text"]: "problem_text",
#     })
#     if "created_at" in C.ISSUE_COLS and C.ISSUE_COLS["created_at"] in issues:
#         issues = issues.rename(columns={C.ISSUE_COLS["created_at"]:"issue_created_at"})
#     if "tags" in C.ISSUE_COLS and C.ISSUE_COLS["tags"] in issues:
#         issues = issues.rename(columns={C.ISSUE_COLS["tags"]:"issue_tags"})

#     comments = comments.rename(columns={
#         C.COMMENT_COLS["issue_id"]: "issue_id",
#         C.COMMENT_COLS["comment_id"]: "comment_id",
#         C.COMMENT_COLS["text"]: "text",
#     })
#     if "author_role" in C.COMMENT_COLS and C.COMMENT_COLS["author_role"] in comments:
#         comments = comments.rename(columns={C.COMMENT_COLS["author_role"]:"author_role"})
#     else:
#         comments["author_role"] = "unknown"
#     if "created_at" in C.COMMENT_COLS and C.COMMENT_COLS["created_at"] in comments:
#         comments = comments.rename(columns={C.COMMENT_COLS["created_at"]:"comment_created_at"})
#     else:
#         comments["comment_created_at"] = None

#     # Clean whitespace
#     for col in ["problem_text","text","author_role"]:
#         if col in issues: issues[col] = issues[col].astype(str).fillna("").str.strip()
#         if col in comments: comments[col] = comments[col].astype(str).fillna("").str.strip()

#     return issues, comments

# def normalise_line(s):
#     s = re.sub(r"\s+", " ", s or "").strip()
#     return s[:1500]  # trim any monster lines

# def build_evidence_block(comments_df):
#     # sort newest first; cap to avoid huge prompts
#     df = comments_df.sort_values(by="comment_created_at", ascending=True, na_position="last")
#     df = df.tail(C.MAX_COMMENTS)

#     lines = []
#     for _, r in df.iterrows():
#         cid = r.get("comment_id","")
#         role = (r.get("author_role","") or "").lower()
#         dt   = str(r.get("comment_created_at","")).split(".")[0]
#         txt  = normalise_line(r.get("text",""))
#         if not txt: continue
#         line = f"{cid} • {role or 'unknown'} • {dt} • {txt}"
#         lines.append(line)
#     return "\n".join(lines)

# def make_prompt(problem_text, evidence_block):
#     prompt = f"""You are a support summarizer.
# Given a helpdesk Problem and evidence comments, produce a short, user-facing resolution in JSON.

# Write style: clear, plain English, no fluff.
# Steps: 2–4 bullets, each starting with an imperative verb, <=20 words.
# Classify solution_type as one of: "User action", "Vendor fix", "Workaround", "Escalate".
# Confidence: "High" | "Medium" | "Low" based on evidence.

# Return only valid JSON with fields:
# - solution_summary
# - steps (array)
# - fallback
# - solution_type
# - tags (array)
# - evidence_comment_ids (array of integers if available)
# - last_verified_date (YYYY-MM-DD)
# - confidence

# Problem:
# {problem_text}

# Evidence comments (id • author_role • date • text):
# {evidence_block}

# Today’s date: {date.today().isoformat()}
# """
#     return prompt

# # ---------- LLM CALLS (optional) ----------
# load_dotenv()

# def call_openai(prompt: str) -> dict:
#     """Uses OpenAI or Azure OpenAI if configured. Returns parsed JSON dict."""
#     provider = (C.LLM_PROVIDER or "none").lower()
#     if provider == "none":
#         return {}

#     if provider == "openai":
#         api_key = os.getenv("OPENAI_API_KEY", "")
#         if not api_key: return {}
#         url = "https://api.openai.com/v1/chat/completions"
#         payload = {
#             "model": C.OPENAI_MODEL,
#             "messages": [
#                 {"role":"system","content":"You only output valid JSON."},
#                 {"role":"user","content": prompt}
#             ],
#             "temperature": 0.2
#         }
#         headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
#         resp = requests.post(url, headers=headers, json=payload, timeout=60)
#         resp.raise_for_status()
#         content = resp.json()["choices"][0]["message"]["content"]
#         return safe_json(content)

#     if provider == "azure":
#         endpoint = os.getenv("AZURE_OPENAI_ENDPOINT","").rstrip("/")
#         api_key  = os.getenv("AZURE_OPENAI_API_KEY","")
#         deploy   = os.getenv("AZURE_OPENAI_DEPLOYMENT", C.OPENAI_MODEL)
#         if not (endpoint and api_key and deploy): return {}
#         url = f"{endpoint}/openai/deployments/{deploy}/chat/completions?api-version=2024-02-15-preview"
#         payload = {
#             "messages":[
#                 {"role":"system","content":"You only output valid JSON."},
#                 {"role":"user","content": prompt}
#             ],
#             "temperature":0.2
#         }
#         headers = {"api-key": api_key, "Content-Type":"application/json"}
#         resp = requests.post(url, headers=headers, json=payload, timeout=60)
#         resp.raise_for_status()
#         content = resp.json()["choices"][0]["message"]["content"]
#         return safe_json(content)

#     return {}

# def safe_json(s: str) -> dict:
#     s = s.strip()
#     # strip code fences if any
#     s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
#     try:
#         return json.loads(s)
#     except Exception:
#         # Last resort: try to extract the first {...}
#         match = re.search(r"\{.*\}", s, flags=re.S)
#         if match:
#             try: return json.loads(match.group(0))
#             except: pass
#     return {}

# # ------------------------------------------

# def main():
#     issues, comments = load_tables()

#     prompts_rows = []
#     kb_rows = []
#     for _, issue in tqdm(issues.iterrows(), total=len(issues), desc="Processing POIs"):
#         iid = issue["issue_id"]
#         problem = normalise_line(issue["problem_text"])
#         subset = comments[comments["issue_id"] == iid]
#         evidence = build_evidence_block(subset)
#         prompt = make_prompt(problem, evidence)

#         prompts_rows.append({"issue_id": iid, "prompt": prompt})

#         # Optional LLM call
#         result = call_openai(prompt)
#         if result:
#             kb_rows.append({
#                 "issue_id": iid,
#                 "problem_text": problem,
#                 "solution_summary": result.get("solution_summary",""),
#                 "steps_solution": json.dumps(result.get("steps",[]), ensure_ascii=False),
#                 "fallback": result.get("fallback",""),
#                 "solution_type": result.get("solution_type",""),
#                 "tags": json.dumps(result.get("tags",[]), ensure_ascii=False),
#                 "evidence_comment_ids": json.dumps(result.get("evidence_comment_ids",[])),
#                 "last_verified_date": result.get("last_verified_date",""),
#                 "confidence": result.get("confidence","")
#             })

#     # Save prompts for inspection
#     pd.DataFrame(prompts_rows).to_csv(OUT_DIR / "prompts.csv", index=False)

#     # Save KB (only if LLM results exist)
#     if kb_rows:
#         pd.DataFrame(kb_rows).to_csv(OUT_DIR / "kb.csv", index=False)
#         print("KB written to out/kb.csv")
#     else:
#         print("No LLM provider configured (or no key). Prompts saved to out/prompts.csv")

# if __name__ == "__main__":
#     main()











# import os, json, re, pathlib
# from datetime import date
# import pandas as pd
# from tqdm import tqdm
# import config as C

# OUT_DIR = pathlib.Path("out")
# OUT_DIR.mkdir(exist_ok=True)

# # ---------------- Utils ----------------
# def _exists(path: str):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Excel file not found:\n  {path}\n"
#                                 "Check EXCEL_PATH in config.py and ensure the file is closed in Excel.")

# def _norm(s: str) -> str:
#     s = str(s or "")
#     s = re.sub(r"\s+", " ", s).strip()
#     return s[:2000]

# def _sentences(text: str):
#     text = _norm(text)
#     parts = re.split(r"(?<=[.!?])\s+", text)
#     parts = [p.strip(" -•\t") for p in parts if p.strip()]
#     return parts[:6]

# # ---------------- Load ----------------
# def load_tables():
#     _exists(C.EXCEL_PATH)

#     issues = pd.read_excel(C.EXCEL_PATH, sheet_name=C.ISSUES_SHEET, engine="openpyxl")
#     comments = pd.read_excel(C.EXCEL_PATH, sheet_name=C.COMMENTS_SHEET, engine="openpyxl")

#     issues = issues.rename(columns={
#         C.ISSUE_COLS["issue_id"]: "issue_id",
#         C.ISSUE_COLS["problem_text"]: "problem_text",
#     })
#     if "created_at" in C.ISSUE_COLS and C.ISSUE_COLS["created_at"] in issues.columns:
#         issues = issues.rename(columns={C.ISSUE_COLS["created_at"]: "issue_created_at"})
#     else:
#         issues["issue_created_at"] = pd.NaT

#     if "tags" in C.ISSUE_COLS and C.ISSUE_COLS["tags"] in issues.columns:
#         issues = issues.rename(columns={C.ISSUE_COLS["tags"]: "issue_tags"})
#     else:
#         issues["issue_tags"] = ""

#     comments = comments.rename(columns={
#         C.COMMENT_COLS["issue_id"]: "issue_id",
#         C.COMMENT_COLS["comment_id"]: "comment_id",
#         C.COMMENT_COLS["text"]: "text",
#     })
#     if "author_role" in C.COMMENT_COLS and C.COMMENT_COLS["author_role"] in comments.columns:
#         comments = comments.rename(columns={C.COMMENT_COLS["author_role"]: "author_role"})
#     else:
#         comments["author_role"] = "unknown"
#     if "created_at" in C.COMMENT_COLS and C.COMMENT_COLS["created_at"] in comments.columns:
#         comments = comments.rename(columns={C.COMMENT_COLS["created_at"]: "comment_created_at"})
#     else:
#         comments["comment_created_at"] = pd.NaT

#     issues["issue_id"] = issues["issue_id"].astype(str)
#     issues["problem_text"] = issues["problem_text"].astype(str).fillna("").map(_norm)

#     comments["issue_id"] = comments["issue_id"].astype(str)
#     comments["text"] = comments["text"].astype(str).fillna("").map(_norm)
#     comments["author_role"] = comments["author_role"].astype(str).str.lower().fillna("unknown")

#     # Drop blank comments
#     comments = comments[comments["text"].str.len() > 0]
#     return issues, comments

# # ---------------- Heuristics ----------------
# FIX_KEYWORDS = [
#     "pushed to production", "deployed", "patch", "patched", "hotfix",
#     "update", "updated", "upgrade", "install", "reinstall",
#     "rolled out", "released", "publish", "enabled", "configured",
#     "removed", "remove application", "resolved", "fixed", "preventive action added"
# ]
# CLOSURE_WORDS = ["closing this ticket", "issue closed", "resolved", "fixed", "confirmed", "will close this ticket"]
# STATUS_WORDS = [
#     "emailed", "replied", "will", "waiting", "pending", "chaser", "follow up",
#     "rca", "investigating", "scheduled a meeting", "testing", "webhook"
# ]
# FAILURE_CUES = ["still", "didn't", "doesn't", "unable", "not able", "same error", "no change"]

# IMPERATIVE_START = re.compile(
#     r"^(click|go to|open|select|choose|set|enable|disable|update|install|reinstall|restart|reset|clear|verify|submit|navigate|use|change|switch|remove)\b",
#     re.I
# )

# def detect_signals(text: str) -> dict:
#     t = text.lower()
#     return {
#         "fix": any(k in t for k in FIX_KEYWORDS),
#         "closure": any(k in t for k in CLOSURE_WORDS),
#         "status": any(re.search(rf"\b{re.escape(w)}\b", t) for w in STATUS_WORDS),
#         "failure": any(k in t for k in FAILURE_CUES),
#         "imperative": bool(IMPERATIVE_START.search(text)),
#     }

# def score_comment(row) -> tuple[float, dict]:
#     s = detect_signals(row["text"])
#     score = 0.0
#     if s["fix"]: score += 2.0
#     if s["imperative"]: score += 1.4
#     if (row.get("author_role","") or "").startswith("staff"): score += 0.8
#     if s["closure"]: score += 0.8
#     if s["status"]: score -= 0.4
#     if s["failure"]: score -= 0.8
#     # minor recency boost if we ever add timestamps later
#     if pd.notna(row.get("comment_created_at", pd.NaT)):
#         score += 0.1
#     return score, s

# def infer_solution_type(text: str, signals: dict) -> str:
#     t = text.lower()
#     if "pushed to production" in t or "patch" in t or "deployed" in t or "updated secure browser" in t:
#         return "Vendor fix"
#     if signals.get("imperative"):
#         return "User action"
#     if "remove application" in t or "reschedule" in t or "free assessment" in t:
#         return "Workaround"
#     return "Workaround"

# def steps_from_text(text: str, sol_type: str) -> list[str]:
#     t = text.lower()
#     steps = []
#     if sol_type == "Vendor fix":
#         steps = [
#             "Update to the latest client/app or Secure Browser.",
#             "Restart the app or browser and retry the action.",
#             "If still blocked, capture a screenshot and raise a support case."
#         ]
#         if "secure browser" in t:
#             steps[0] = "Update the Secure Browser to the latest build."
#     elif sol_type == "User action":
#         sents = _sentences(text)
#         picked = []
#         for s in sents:
#             if IMPERATIVE_START.search(s) or len(s) <= 120:
#                 picked.append(s)
#             if len(picked) >= 3: break
#         steps = picked or ["Retry the action.", "Restart the app/browser.", "If it persists, raise a support case."]
#     else:
#         steps = [
#             "Retry later or from a different network/device.",
#             "If still blocked, capture a screenshot and open a support ticket.",
#             "Include OS, app/browser version, and exact error text."
#         ]
#     return steps[:4]

# def choose_solution(comments_df: pd.DataFrame):
#     if comments_df.empty:
#         return None

#     scored = []
#     for _, r in comments_df.iterrows():
#         sc, sig = score_comment(r)
#         scored.append((sc, sig, r))
#     scored.sort(key=lambda x: (x[0], str(x[2].get("comment_created_at",""))), reverse=True)

#     # best candidate with positive score
#     top = next((x for x in scored if x[0] > 0.5), None)
#     if top:
#         score, signals, row = top
#         sol_type = infer_solution_type(row["text"], signals)
#         steps = steps_from_text(row["text"], sol_type)
#         summary = row["text"]
#         conf = "High" if signals.get("closure") or signals.get("fix") else "Medium"
#         evidence = []
#         try:
#             evidence = [int(row["comment_id"])] if pd.notna(row["comment_id"]) else []
#         except Exception:
#             pass
#         # add closure evidence if present elsewhere
#         close = next((rr for sc, sg, rr in scored if sg.get("closure")), None)
#         if close is not None:
#             try:
#                 cid = int(close["comment_id"])
#                 if cid not in evidence: evidence.append(cid)
#                 conf = "High"
#             except Exception:
#                 pass
#         return {
#             "solution_summary": summary[:180],
#             "steps": steps,
#             "solution_type": sol_type,
#             "confidence": conf,
#             "evidence_comment_ids": evidence
#         }

#     any_closure = any(sig.get("closure") for _, sig, _ in scored)
#     sol_type = "Vendor fix" if any_closure else "Workaround"
#     steps = steps_from_text("", sol_type)
#     evidence = []
#     if any_closure:
#         try:
#             evidence = [int(r["comment_id"]) for sc, sig, r in scored if sig.get("closure")][:1]
#         except Exception:
#             evidence = []
#     return {
#         "solution_summary": ("Issue resolved by platform update; retry with latest version."
#                              if sol_type=="Vendor fix" else
#                              "Workaround/Retry flow (no explicit fix found)."),
#         "steps": steps,
#         "solution_type": sol_type,
#         "confidence": "Medium" if any_closure else "Low",
#         "evidence_comment_ids": evidence
#     }

# # ---------------- Evidence block (for prompts CSV only) ----------------
# def build_evidence_block(df_comments: pd.DataFrame) -> str:
#     df = df_comments.copy()
#     if "comment_created_at" in df.columns and df["comment_created_at"].notna().any():
#         df = df.sort_values(by="comment_created_at", ascending=True)
#     df = df.tail(C.MAX_COMMENTS)
#     lines = []
#     for _, r in df.iterrows():
#         cid = r.get("comment_id","")
#         role = r.get("author_role","unknown") or "unknown"
#         dt   = str(r.get("comment_created_at","") or "")
#         txt  = _norm(r.get("text",""))
#         if not txt: continue
#         lines.append(f"{cid} • {role} • {dt} • {txt}")
#     return "\n".join(lines)

# def make_user_summary_prompt(problem_text: str, evidence_block: str) -> str:
#     return f"""You are a support summarizer.
# Given a helpdesk Problem and evidence comments, produce a short, user-facing resolution in JSON.
# (Template kept for future LLM use.)

# Problem:
# {problem_text}

# Evidence comments (id • author_role • date • text):
# {evidence_block}

# Today’s date: {date.today().isoformat()}
# """

# # ---------------- Main ----------------
# def main():
#     print("Loading Excel…")
#     issues, comments = load_tables()
#     print(f"Issues: {len(issues)} | Comments: {len(comments)}")

#     prompts_rows = []
#     kb_rows = []

#     grouped = comments.groupby("issue_id", sort=False)

#     for _, issue in tqdm(issues.iterrows(), total=len(issues), desc="Processing POIs"):
#         iid = issue["issue_id"]
#         problem = _norm(issue["problem_text"])
#         cmts = grouped.get_group(iid) if iid in grouped.groups else pd.DataFrame(columns=comments.columns)

#         # Keep prompts CSV (handy if you later enable LLM)
#         evidence = build_evidence_block(cmts)
#         prompts_rows.append({"issue_id": iid, "prompt": make_user_summary_prompt(problem, evidence)})

#         # Heuristic solution selection
#         choice = choose_solution(cmts)
#         if choice:
#             tags = []
#             low_problem = problem.lower()
#             low_evidence = evidence.lower()
#             if "photo verification" in low_problem or "photo verification" in low_evidence:
#                 tags.append("Photo Verification")
#             if "secure browser" in low_problem or "secure browser" in low_evidence:
#                 tags.append("Secure Browser")
#             if "instruction page" in low_problem:
#                 tags.append("Instruction Page")

#             kb_rows.append({
#                 "issue_id": iid,
#                 "problem_text": problem,
#                 "solution_summary": choice["solution_summary"],
#                 "steps_solution": json.dumps(choice["steps"], ensure_ascii=False),
#                 "fallback": "If the steps fail, capture a screenshot and open a support case with OS, version and exact error.",
#                 "solution_type": choice["solution_type"],
#                 "tags": json.dumps(tags, ensure_ascii=False),
#                 "evidence_comment_ids": json.dumps(choice["evidence_comment_ids"]),
#                 "last_verified_date": date.today().isoformat(),
#                 "confidence": choice["confidence"]
#             })
#         else:
#             kb_rows.append({
#                 "issue_id": iid,
#                 "problem_text": problem,
#                 "solution_summary": "No solution comments available.",
#                 "steps_solution": json.dumps([
#                     "Retry the action.",
#                     "Restart app/browser and try again.",
#                     "Escalate with screenshot and details if it persists."
#                 ], ensure_ascii=False),
#                 "fallback": "Open a support case with logs and screenshots.",
#                 "solution_type": "Escalate",
#                 "tags": json.dumps([], ensure_ascii=False),
#                 "evidence_comment_ids": "[]",
#                 "last_verified_date": date.today().isoformat(),
#                 "confidence": "Low"
#             })

#     pd.DataFrame(prompts_rows).to_csv(OUT_DIR / "prompts.csv", index=False, encoding="utf-8-sig")
#     pd.DataFrame(kb_rows).to_csv(OUT_DIR / "kb_heuristic.csv", index=False, encoding="utf-8-sig")
#     print("Saved prompts → out/prompts.csv")
#     print("Saved heuristic KB → out/kb_heuristic.csv")

# if __name__ == "__main__":
#     main()




















import os, json, re, pathlib
from datetime import date
import pandas as pd
from tqdm import tqdm
import config as C

OUT_DIR = pathlib.Path("out")
OUT_DIR.mkdir(exist_ok=True)

# ---------------- Utils ----------------
def _exists(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found:\n  {path}\n"
                                "Check EXCEL_PATH in config.py and ensure the file is closed in Excel.")

def _norm(s: str) -> str:
    s = str(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s[:2000]

def _sentences(text: str):
    text = _norm(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip(" -•\t") for p in parts if p.strip()]
    return parts[:6]

# ---------------- Load ----------------
def load_tables():
    _exists(C.EXCEL_PATH)

    issues = pd.read_excel(C.EXCEL_PATH, sheet_name=C.ISSUES_SHEET, engine="openpyxl")
    comments = pd.read_excel(C.EXCEL_PATH, sheet_name=C.COMMENTS_SHEET, engine="openpyxl")

    issues = issues.rename(columns={
        C.ISSUE_COLS["issue_id"]: "issue_id",
        C.ISSUE_COLS["problem_text"]: "problem_text",
    })
    if "created_at" in C.ISSUE_COLS and C.ISSUE_COLS["created_at"] in issues.columns:
        issues = issues.rename(columns={C.ISSUE_COLS["created_at"]: "issue_created_at"})
    else:
        issues["issue_created_at"] = pd.NaT

    if "tags" in C.ISSUE_COLS and C.ISSUE_COLS["tags"] in issues.columns:
        issues = issues.rename(columns={C.ISSUE_COLS["tags"]: "issue_tags"})
    else:
        issues["issue_tags"] = ""

    comments = comments.rename(columns={
        C.COMMENT_COLS["issue_id"]: "issue_id",
        C.COMMENT_COLS["comment_id"]: "comment_id",
        C.COMMENT_COLS["text"]: "text",
    })
    if "author_role" in C.COMMENT_COLS and C.COMMENT_COLS["author_role"] in comments.columns:
        comments = comments.rename(columns={C.COMMENT_COLS["author_role"]: "author_role"})
    else:
        comments["author_role"] = "unknown"
    if "created_at" in C.COMMENT_COLS and C.COMMENT_COLS["created_at"] in comments.columns:
        comments = comments.rename(columns={C.COMMENT_COLS["created_at"]: "comment_created_at"})
    else:
        comments["comment_created_at"] = pd.NaT

    issues["issue_id"] = issues["issue_id"].astype(str)
    issues["problem_text"] = issues["problem_text"].astype(str).fillna("").map(_norm)

    comments["issue_id"] = comments["issue_id"].astype(str)
    comments["text"] = comments["text"].astype(str).fillna("").map(_norm)
    comments["author_role"] = comments["author_role"].astype(str).str.lower().fillna("unknown")

    # Drop blank comments
    comments = comments[comments["text"].str.len() > 0]
    return issues, comments

# ---------------- Heuristics ----------------
FIX_KEYWORDS = [
    "pushed to production", "deployed", "patch", "patched", "hotfix",
    "update", "updated", "upgrade", "install", "reinstall",
    "rolled out", "released", "publish", "enabled", "configured",
    "removed", "remove application", "resolved", "fixed", "preventive action added"
]
CLOSURE_WORDS = ["closing this ticket", "issue closed", "resolved", "fixed", "confirmed", "will close this ticket"]
STATUS_WORDS = [
    "emailed", "replied", "will", "waiting", "pending", "chaser", "follow up",
    "rca", "investigating", "scheduled a meeting", "testing", "webhook"
]
FAILURE_CUES = ["still", "didn't", "doesn't", "unable", "not able", "same error", "no change"]

IMPERATIVE_START = re.compile(
    r"^(click|go to|open|select|choose|set|enable|disable|update|install|reinstall|restart|reset|clear|verify|submit|navigate|use|change|switch|remove)\b",
    re.I
)

def detect_signals(text: str) -> dict:
    t = text.lower()
    return {
        "fix": any(k in t for k in FIX_KEYWORDS),
        "closure": any(k in t for k in CLOSURE_WORDS),
        "status": any(re.search(rf"\b{re.escape(w)}\b", t) for w in STATUS_WORDS),
        "failure": any(k in t for k in FAILURE_CUES),
        "imperative": bool(IMPERATIVE_START.search(text)),
    }

def score_comment(row) -> tuple[float, dict]:
    s = detect_signals(row["text"])
    score = 0.0
    if s["fix"]: score += 2.0
    if s["imperative"]: score += 1.4
    if (row.get("author_role","") or "").startswith("staff"): score += 0.8
    if s["closure"]: score += 0.8
    if s["status"]: score -= 0.4
    if s["failure"]: score -= 0.8
    # minor recency boost if we ever add timestamps later
    if pd.notna(row.get("comment_created_at", pd.NaT)):
        score += 0.1
    return score, s

def infer_solution_type(text: str, signals: dict) -> str:
    t = text.lower()
    if "pushed to production" in t or "patch" in t or "deployed" in t or "updated secure browser" in t:
        return "Vendor fix"
    if signals.get("imperative"):
        return "User action"
    if "remove application" in t or "reschedule" in t or "free assessment" in t:
        return "Workaround"
    return "Workaround"

def steps_from_text(text: str, sol_type: str) -> list[str]:
    t = text.lower()
    steps = []
    if sol_type == "Vendor fix":
        steps = [
            "Update to the latest client/app or Secure Browser.",
            "Restart the app or browser and retry the action.",
            "If still blocked, capture a screenshot and raise a support case."
        ]
        if "secure browser" in t:
            steps[0] = "Update the Secure Browser to the latest build."
    elif sol_type == "User action":
        sents = _sentences(text)
        picked = []
        for s in sents:
            if IMPERATIVE_START.search(s) or len(s) <= 120:
                picked.append(s)
            if len(picked) >= 3: break
        steps = picked or ["Retry the action.", "Restart the app/browser.", "If it persists, raise a support case."]
    else:
        steps = [
            "Retry later or from a different network/device.",
            "If still blocked, capture a screenshot and open a support ticket.",
            "Include OS, app/browser version, and exact error text."
        ]
    return steps[:4]

def choose_solution(comments_df: pd.DataFrame):
    if comments_df.empty:
        return None

    scored = []
    for _, r in comments_df.iterrows():
        sc, sig = score_comment(r)
        scored.append((sc, sig, r))
    scored.sort(key=lambda x: (x[0], str(x[2].get("comment_created_at",""))), reverse=True)

    # best candidate with positive score
    top = next((x for x in scored if x[0] > 0.5), None)
    if top:
        score, signals, row = top
        sol_type = infer_solution_type(row["text"], signals)
        steps = steps_from_text(row["text"], sol_type)
        summary = row["text"]
        conf = "High" if signals.get("closure") or signals.get("fix") else "Medium"
        evidence = []
        try:
            evidence = [int(row["comment_id"])] if pd.notna(row["comment_id"]) else []
        except Exception:
            pass
        # add closure evidence if present elsewhere
        close = next((rr for sc, sg, rr in scored if sg.get("closure")), None)
        if close is not None:
            try:
                cid = int(close["comment_id"])
                if cid not in evidence: evidence.append(cid)
                conf = "High"
            except Exception:
                pass
        return {
            "solution_summary": summary[:180],
            "steps": steps,
            "solution_type": sol_type,
            "confidence": conf,
            "evidence_comment_ids": evidence
        }

    any_closure = any(sig.get("closure") for _, sig, _ in scored)
    sol_type = "Vendor fix" if any_closure else "Workaround"
    steps = steps_from_text("", sol_type)
    evidence = []
    if any_closure:
        try:
            evidence = [int(r["comment_id"]) for sc, sig, r in scored if sig.get("closure")][:1]
        except Exception:
            evidence = []
    return {
        "solution_summary": ("Issue resolved by platform update; retry with latest version."
                             if sol_type=="Vendor fix" else
                             "Workaround/Retry flow (no explicit fix found)."),
        "steps": steps,
        "solution_type": sol_type,
        "confidence": "Medium" if any_closure else "Low",
        "evidence_comment_ids": evidence
    }

# ---------------- Evidence block (for prompts CSV only) ----------------
def build_evidence_block(df_comments: pd.DataFrame) -> str:
    df = df_comments.copy()
    if "comment_created_at" in df.columns and df["comment_created_at"].notna().any():
        df = df.sort_values(by="comment_created_at", ascending=True)
    df = df.tail(C.MAX_COMMENTS)
    lines = []
    for _, r in df.iterrows():
        cid = r.get("comment_id","")
        role = r.get("author_role","unknown") or "unknown"
        dt   = str(r.get("comment_created_at","") or "")
        txt  = _norm(r.get("text",""))
        if not txt: continue
        lines.append(f"{cid} • {role} • {dt} • {txt}")
    return "\n".join(lines)

def make_user_summary_prompt(problem_text: str, evidence_block: str) -> str:
    return f"""You are a support summarizer.
Given a helpdesk Problem and evidence comments, produce a short, user-facing resolution in JSON.
(Template kept for future LLM use.)

Problem:
{problem_text}

Evidence comments (id • author_role • date • text):
{evidence_block}

Today’s date: {date.today().isoformat()}
"""

# ---------------- Main ----------------
def main():
    print("Loading Excel…")
    issues, comments = load_tables()
    print(f"Issues: {len(issues)} | Comments: {len(comments)}")

    prompts_rows = []
    kb_rows = []

    grouped = comments.groupby("issue_id", sort=False)

    for _, issue in tqdm(issues.iterrows(), total=len(issues), desc="Processing POIs"):
        iid = issue["issue_id"]
        problem = _norm(issue["problem_text"])
        cmts = grouped.get_group(iid) if iid in grouped.groups else pd.DataFrame(columns=comments.columns)

        # Keep prompts CSV (handy if you later enable LLM)
        evidence = build_evidence_block(cmts)
        prompts_rows.append({"issue_id": iid, "prompt": make_user_summary_prompt(problem, evidence)})

        # Heuristic solution selection
        choice = choose_solution(cmts)
        if choice:
            tags = []
            low_problem = problem.lower()
            low_evidence = evidence.lower()
            if "photo verification" in low_problem or "photo verification" in low_evidence:
                tags.append("Photo Verification")
            if "secure browser" in low_problem or "secure browser" in low_evidence:
                tags.append("Secure Browser")
            if "instruction page" in low_problem:
                tags.append("Instruction Page")

            kb_rows.append({
                "issue_id": iid,
                "problem_text": problem,
                "solution_summary": choice["solution_summary"],
                "steps_solution": json.dumps(choice["steps"], ensure_ascii=False),
                "fallback": "If the steps fail, capture a screenshot and open a support case with OS, version and exact error.",
                "solution_type": choice["solution_type"],
                "tags": json.dumps(tags, ensure_ascii=False),
                "evidence_comment_ids": json.dumps(choice["evidence_comment_ids"]),
                "last_verified_date": date.today().isoformat(),
                "confidence": choice["confidence"]
            })
        else:
            kb_rows.append({
                "issue_id": iid,
                "problem_text": problem,
                "solution_summary": "No solution comments available.",
                "steps_solution": json.dumps([
                    "Retry the action.",
                    "Restart app/browser and try again.",
                    "Escalate with screenshot and details if it persists."
                ], ensure_ascii=False),
                "fallback": "Open a support case with logs and screenshots.",
                "solution_type": "Escalate",
                "tags": json.dumps([], ensure_ascii=False),
                "evidence_comment_ids": "[]",
                "last_verified_date": date.today().isoformat(),
                "confidence": "Low"
            })

    pd.DataFrame(prompts_rows).to_csv(OUT_DIR / "prompts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(kb_rows).to_csv(OUT_DIR / "kb_heuristic.csv", index=False, encoding="utf-8-sig")
    print("Saved prompts → out/prompts.csv")
    print("Saved heuristic KB → out/kb_heuristic.csv")

if __name__ == "__main__":
    main()
