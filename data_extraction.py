import json
import re
from pathlib import Path
from datetime import date

import pandas as pd

KB_PATH = Path("out/kb_heuristic.csv")
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _compose_solution(summary: str, steps_json: str, fallback: str) -> str:
    """
    Build a single readable solution block:
      <summary>
      1) step
      2) step
      If this still fails, <fallback>
    """
    parts = []

    summary = (summary or "").strip()
    if summary:
        parts.append(summary)

    # steps are stored as JSON string in kb_heuristic.csv
    steps = []
    try:
        steps = json.loads(steps_json or "[]")
        if isinstance(steps, dict):  # in case it's a dict accidentally.
            steps = list(steps.values())
    except Exception:
        steps = []

    if steps:
        for i, s in enumerate(steps, 1):
            s = _norm(s)
            if s:
                parts.append(f"{i}. {s}")

    fb = _norm(fallback or "")
    if fb:
        # keep your consistent phrasing
        parts.append(f"If this still fails, {fb}")

    return "\n".join(parts).strip()

def load_kb() -> pd.DataFrame:
    if not KB_PATH.exists():
        raise SystemExit(
            "KB not found at out/kb_heuristic.csv.\n"
            "Run:  python main.py\n"
            "so it can generate the knowledge base first."
        )
    df = pd.read_csv(KB_PATH)
    # ensure expected columns exist (lenient)
    for col in ["issue_id", "problem_text", "solution_summary", "steps_solution", "fallback", "confidence", "tags"]:
        if col not in df.columns:
            df[col] = ""
    return df

def build_export(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        issue_id = r.get("issue_id", "")
        problem = r.get("problem_text", "")
        summary = r.get("solution_summary", "")
        steps_json = r.get("steps_solution", "[]")
        fallback = r.get("fallback", "")
        confidence = r.get("confidence", "")
        tags = r.get("tags", "")

        solution_full = _compose_solution(summary, steps_json, fallback)

        # keep both the composed text and the raw pieces
        rows.append({
            "issue_id": issue_id,
            "problem_text": problem,
            "solution_full": solution_full,
            "solution_summary": summary,
            "steps_solution": steps_json,   # still as JSON string
            "fallback": fallback,
            "confidence": confidence,
            "tags": tags,
            "export_date": date.today().isoformat(),
        })

    return pd.DataFrame(rows)

def save_all(df_out: pd.DataFrame):
    csv_path = OUT_DIR / "problems_solutions.csv"
    xlsx_path = OUT_DIR / "problems_solutions.xlsx"
    jsonl_path = OUT_DIR / "problems_solutions.jsonl"

    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        df_out.to_excel(xlsx_path, index=False, engine="openpyxl")
    except Exception as e:
        print(f"[warn] Could not write XLSX ({e}). CSV/JSONL still created.")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df_out.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saved → {csv_path}")
    print(f"Saved → {xlsx_path}")
    print(f"Saved → {jsonl_path}")

def main():
    df = load_kb()
    df_out = build_export(df)
    save_all(df_out)

if __name__ == "__main__":
    main()
