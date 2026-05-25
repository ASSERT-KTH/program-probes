from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyReducesFailingProbe(ProbeAdapter):
    """Label: whether each edit has fewer failing tests than the original baseline.

    The baseline edit (cmd_idx=-1) represents the repository state before any
    agent edit; it receives *None* because there is no prior state to compare
    against.  Every subsequent edit receives *True* if its failing-test count
    is strictly less than the baseline count, *False* otherwise.  An edit
    whose ``test_results`` is *None* also produces *None*.
    """

    name = "currently_reduces_failing"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        labels: list[bool | None] = []
        baseline_failing: int | None = None
        for edit in ctx.edit_history:
            tr = edit.test_results
            if tr is None:
                labels.append(None)
            elif baseline_failing is None:
                # This is the baseline edit — store its count, no label yet.
                baseline_failing = len(tr.get("failed", []))
                labels.append(None)
            else:
                curr_failing = len(tr.get("failed", []))
                labels.append(curr_failing < baseline_failing)
        return labels
