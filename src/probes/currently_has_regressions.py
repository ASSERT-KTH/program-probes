from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyHasRegressionsProbe(ProbeAdapter):
    """Label: whether each edit introduced regressions.

    At each edit we check whether any test name appearing in
    ``test_results["failed"]`` was in the *previous* edit's
    ``test_results["passed"]``.  If so, the label is *True*.

    The first edit has no baseline and is always *False*.  An edit whose
    ``test_results`` is *None* produces *None*.
    """

    name = "currently_has_regressions"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        labels: list[bool | None] = []
        prev_passed: set[str] = set()
        for edit in ctx.edit_history:
            tr = edit.test_results
            if tr is None:
                labels.append(None)
            elif not prev_passed:
                labels.append(False)
            else:
                curr_failed = set(tr.get("failed", []))
                labels.append(not curr_failed.isdisjoint(prev_passed))
            if tr is not None:
                prev_passed = set(tr.get("passed", []))
        return labels
