"""Brief PO helper: the task prompt text (first user message) per task, for the TF-IDF prompt-only baseline.

Reads one generation file per task (all runs of a task share the prompt; checked on a second run when
present) and writes tmp/po_texts_<bench>_<run>.json = {task: text}.
Usage: uvx python@3.12 scripts/rebuttal_prompt_texts.py
"""
import json
import re
from pathlib import Path

for bench, run in [("swebench", "laguna_xs2_full"), ("swebench", "qwen36_35b_a3b_full"),
                   ("swebench_pro", "laguna_xs2_full"), ("swebench_pro", "qwen36_35b_a3b_full")]:
    by_task: dict[str, list[Path]] = {}
    for f in sorted(Path("generations", bench, run).glob("*.json")):
        by_task.setdefault(re.sub(r"_run\d+$", "", f.stem), []).append(f)
    texts, checked, mismatch = {}, 0, 0
    for t, fs in by_task.items():
        def first_user(p: Path) -> str:
            m = json.loads(p.read_text())["messages"]
            return next(x["content"] for x in m if x["role"] == "user")
        texts[t] = first_user(fs[0])
        if len(fs) > 1 and checked < 50:
            checked += 1
            mismatch += first_user(fs[1]) != texts[t]
    Path("tmp", f"po_texts_{bench}_{run}.json").write_text(json.dumps(texts))
    print(bench, run, "tasks", len(texts), "files", sum(map(len, by_task.values())),
          "second-run check", checked, "mismatch", mismatch, flush=True)
