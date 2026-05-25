from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyHasRegressionsProbe(ProbeAdapter):
    """Label: whether each edit introduced regressions vs. the original baseline.

    The baseline edit (cmd_idx=-1) represents the repository state before any
    agent edit; it receives *None* because there is no prior state to compare
    against.  Every subsequent edit receives *True* if any test that was
    passing at baseline is now failing, *False* otherwise.  An edit whose
    ``test_results`` is *None* also produces *None*.
    """

    name = "currently_has_regressions"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        labels: list[bool | None] = []
        baseline_passed: set[str] | None = None
        for edit in ctx.edit_history:
            tr = edit.test_results
            if tr is None:
                labels.append(None)
            elif baseline_passed is None:
                # This is the baseline edit — store passing tests.
                # No edit has been made yet, so no regressions are possible.
                baseline_passed = set(tr.get("passed", []))
                labels.append(False)
            else:
                curr_failed = set(tr.get("failed", []))
                labels.append(not curr_failed.isdisjoint(baseline_passed))
        return labels
