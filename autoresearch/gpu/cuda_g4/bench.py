"""G4 entrypoint using the unchanged shared correctness/timing evaluator."""
from pathlib import Path
import argparse
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoresearch.llm_common import TorchBackend, evaluate, load_candidate


class G4Backend(TorchBackend):
    def __init__(self, smoke):
        # The shared A100 backend otherwise rejects this GPU. Keep all evaluation
        # checks unchanged and enforce G4 identity here, including for smoke runs.
        super().__init__(smoke=True)
        capability = self.t.cuda.get_device_capability()
        if "RTX PRO 6000" not in self.name or "Blackwell" not in self.name or capability != (12, 0):
            raise RuntimeError(f"G4 SM120 required; found {self.name}, {capability}")
        self.versions["cuda"] = self.t.version.cuda


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = dict(status="error", correct=False, scored=False, phase=args.phase)
    try:
        result = evaluate(G4Backend(args.smoke), load_candidate(Path(__file__).with_name("model.py")),
                          args.phase, args.smoke)
    except AssertionError as e:
        result.update(status="incorrect", error=str(e))
        traceback.print_exc()
    except Exception as e:
        result.update(error=str(e))
        traceback.print_exc()
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print("result_json: " + json.dumps(result))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
