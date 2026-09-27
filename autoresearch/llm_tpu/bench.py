"""Fixed evaluator. During research edit model.py only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autoresearch.llm_common import main

if __name__ == "__main__":
    raise SystemExit(main("tpu", Path(__file__).with_name("model.py")))
