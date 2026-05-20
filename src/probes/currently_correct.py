from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCorrectProbe(ProbeAdapter):
    """Label: whether the code state at each edit passes all evaluation tests.

    Returns one label per EditEvent.  If ``test_results["resolved"]`` is present
    it takes precedence; otherwise falls back to ``len(test_results["failed"]) == 0``.
    A ``test_results`` of *None* (eval could not be parsed) produces a *None* label.
    """

    name = "currently_correct"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        labels: list[bool | None] = []
        for edit in ctx.edit_history:
            tr = edit.test_results
            if tr is None:
                labels.append(None)
            elif "resolved" in tr:
                labels.append(tr["resolved"])
            else:
                labels.append(len(tr.get("failed", [])) == 0)
        return labels
