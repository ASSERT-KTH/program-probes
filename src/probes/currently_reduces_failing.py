from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyReducesFailingProbe(ProbeAdapter):
    """Label: whether each edit reduced the number of failing tests.

    At each edit we compare ``len(test_results["failed"])`` with the count at
    the *previous* edit.  If the count decreased the label is *True*; if it
    stayed the same or increased the label is *False*.

    The first edit has no baseline and is always *None*.  An edit whose
    ``test_results`` is *None* also produces *None*.
    """

    name = "currently_reduces_failing"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        labels: list[bool | None] = []
        prev_failing: int | None = None
        for edit in ctx.edit_history:
            tr = edit.test_results
            if tr is None or prev_failing is None:
                labels.append(None)
            else:
                curr_failing = len(tr.get("failed", []))
                labels.append(curr_failing < prev_failing)
            if tr is not None:
                prev_failing = len(tr.get("failed", []))
        return labels
