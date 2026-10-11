"""Fixed evaluator. During research edit only model.py, swiglu.cu, and bindings.cpp."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoresearch.llm_common import main

if __name__ == "__main__":
    raise SystemExit(main("gpu", Path(__file__).with_name("model.py")))
