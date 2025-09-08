import sys, subprocess, json, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
 
def main(_: str) -> list:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    subprocess.check_call([PY, str(ROOT / "data_extraction.py"), "--new-only", "--out", tmp], cwd=ROOT)
    with open(tmp, "r", encoding="utf-8") as f:
        return json.load(f)  # must be list[dict]