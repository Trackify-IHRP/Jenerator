import sys, subprocess
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
 
def main(issue_key: str) -> str:
    subprocess.check_call([PY, str(ROOT / "main.py"), "--issue", issue_key], cwd=ROOT)
    return f"main.py done for {issue_key}"