from src.probes.base import ProbeAdapter, TrajectoryContext


class CurrentlyCorrectSweProbe(ProbeAdapter):
    """Dynamic probe: True at each edit if all task-defined tests passed at that step."""

    name = "currently_correct_swe"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        result = []
        for e in ctx.edit_history:
            if e.test_results is None:
                result.append(None)
            else:
                resolved = e.test_results.get("resolved")
                result.append(bool(resolved) if resolved is not None else None)
        return result
