"""Brief PO cache check: where the first cached token of turn 1 sits after the prompt (one generation file per task).

Writes paper_rebuttal_candidate/prompt_only/cache_check.json. Usage: uvx python@3.12 scripts/rebuttal_prompt_cache_check.py
"""
import json, re, statistics
from pathlib import Path
out = {}
for bench, run in [("swebench", "laguna_xs2_full"), ("swebench", "qwen36_35b_a3b_full"),
                   ("swebench_pro", "laguna_xs2_full"), ("swebench_pro", "qwen36_35b_a3b_full")]:
    seen, gaps, starts, think = set(), [], [], 0
    for f in sorted(Path("generations", bench, run).glob("*.json")):
        t = re.sub(r"_run\d+$", "", f.stem)
        if t in seen: continue
        d = json.loads(f.read_text()); segs = d["tokenization"]["segments"]; msgs = d["messages"]
        a0 = next((j for j, m in enumerate(msgs) if m["role"] == "assistant"), None)
        s = next((s for s in segs if s["message_idx"] == a0), None)
        if s is None: continue
        seen.add(t)
        u = max(x["end_token"] for x in segs if x["message_idx"] < a0)
        gaps.append(s["start_token"] - u); starts.append(s["start_token"])
        txt = d["tokenization"]["text"]; i = txt.find(msgs[a0]["content"])
        think += "</think>" in txt[max(0, i - 3000):i]
    q = lambda v: [statistics.quantiles(v, n=20)[0], statistics.median(v), statistics.quantiles(v, n=20)[-1], min(v), max(v)]
    out[f"{bench}/{run}"] = {"n_tasks_checked": len(gaps), "tokens_between_prompt_end_and_first_cached_token_p5_med_p95_min_max": q(gaps),
                             "first_cached_token_position_p5_med_p95_min_max": q(starts), "n_with_think_block_before_content": think}
    print(bench, run, out[f"{bench}/{run}"], flush=True)
Path("paper_rebuttal_candidate/prompt_only/cache_check.json").write_text(json.dumps(out, indent=1))
