"""Brief E9 helper: per-trajectory texts and command classes for the text baselines.

For every generation file of the four runs, writes one JSON line to tmp/e9_text_<bench>_<run>.jsonl.gz:
  sid      sample id
  n_asst   number of assistant messages (= turns; turn t is the t-th assistant message)
  msgs     [[role, content], ...] of all messages in order (system, task prompt, assistant, observations)
  asst_at  message index of each assistant turn
  runs     per command_history entry: classify(command)["runs_project"] (AE0 primary rule)
  strict   per command_history entry: classify(command)["strict"]
Usage: uv run python scripts/rebuttal_text_extract.py   (CPU job; pure Python)
"""
from __future__ import annotations

import gzip
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from after_edit_cmd_filter import classify  # noqa: E402

RUNS = [("swebench", "laguna_xs2_full"), ("swebench", "qwen36_35b_a3b_full"),
        ("swebench_pro", "laguna_xs2_full"), ("swebench_pro", "qwen36_35b_a3b_full")]


def one(run: tuple[str, str]) -> str:
    bench, name = run
    out = Path("tmp", f"e9_text_{bench}_{name}.jsonl.gz")
    n = n_cmd = n_runs = 0
    with gzip.open(out, "wt") as fo:
        for f in sorted(Path("generations", bench, name).glob("*.json")):
            g = json.loads(f.read_text())
            msgs = [[m["role"], m.get("content") or ""] for m in g["messages"]]
            asst_at = [i for i, (r, _) in enumerate(msgs) if r == "assistant"]
            cls = [classify(c["command"]) for c in g.get("command_history", [])]
            fo.write(json.dumps({"sid": f.stem, "n_asst": len(asst_at), "msgs": msgs, "asst_at": asst_at,
                                 "runs": [c["runs_project"] for c in cls], "strict": [c["strict"] for c in cls]}) + "\n")
            n += 1
            n_cmd += len(cls)
            n_runs += sum(c["runs_project"] for c in cls)
            del g
    return f"{bench}/{name}: {n} trajectories, {n_cmd} commands, {n_runs} run something (runs_project)"


if __name__ == "__main__":
    with Pool(4) as p:
        for s in p.imap_unordered(one, RUNS):
            print(s, flush=True)
