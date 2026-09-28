---
license: cc-by-4.0
---

# latent-programming-horizons-trajs

Agent trajectories and per-edit correctness labels from the
[program-probes](https://github.com/ASSERT-KTH/program-probes) project, which
measures whether a language model's internal hidden states linearly predict
properties of its own agentic output (e.g. "does the code currently compile?")
before those properties are realised.

Each trajectory is a run of a coding agent (mini-SWE-agent) attempting a
SWE-bench (Verified or Pro) instance. This dataset contains the raw
transcripts and labels used to train and evaluate the probes in the paper —
it does not include the extracted hidden-state tensors or trained probes
themselves, since those are reproducible from these trajectories plus the
generating model.

## Layout

```
<task>/<run_id>/<instance_id>[_runNN].json          agent trajectory
<task>/<run_id>/labels/<instance_id>[_runNN]_labels.json   per-edit labels
```

`task` is `swebench` (SWE-bench Verified) or `swebench_pro` (SWE-bench Pro).
`run_id` identifies the generating model/config, e.g. `laguna_xs2_full` or
`qwen36_35b_a3b_full`. Instances with multiple independent samples are
suffixed `_run01`, `_run02`, ... ; the unsuffixed file is the first sample.

### Trajectory format (`schema_version: program-probes.agent_trajectory.v2`)

| Field | Description |
|---|---|
| `task` | The SWE-bench issue text given to the agent. |
| `result` | Agent's final exit status and submission. |
| `model` | Served model name / tokenizer used to generate the trajectory. |
| `messages` | Full chat transcript (system/user/assistant/tool messages). |
| `command_history` | Sequence of shell commands the agent issued. |
| `metadata` | `instance_id`, `repo`, `base_commit`, `outcome` (resolved or not), and the raw sandbox eval log. |
| `tokenization` | Token offsets aligning messages to the tokenized sequence, used for hidden-state extraction. |

### Label format

Each `_labels.json` mirrors the corresponding trajectory and contains one
entry per edit:

```json
{
  "instance_id": "astropy__astropy-12907",
  "edits": [
    {"cmd_idx": -1, "compiles": true, "test_results": {"resolved": false, ...}},
    {"cmd_idx": 9,  "compiles": true, "test_results": {"resolved": true,  ...}}
  ]
}
```

`cmd_idx` indexes into the trajectory's `command_history`; `cmd_idx = -1` is
the clean-checkout baseline used as the comparison point for delta probes
(`currently_reduces_failing`, `currently_has_regressions`).

## Source

Generated and labeled with the pipeline in
[ASSERT-KTH/program-probes](https://github.com/ASSERT-KTH/program-probes) —
see that repo's README for how these trajectories were produced and how to
reproduce hidden-state extraction and probe training from them.

## License

CC-BY-4.0.
