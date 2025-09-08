import sys, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
 
def main(_: str) -> str:
    subprocess.check_call([PY, str(ROOT / "test_search.py"), "train"], cwd=ROOT)
    return "train done"