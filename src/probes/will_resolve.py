from src.probes.base import ProbeAdapter, TrajectoryContext


class WillResolveProbe(ProbeAdapter):
    """Dynamic probe: every captured step is labeled with the trajectory outcome."""

    name = "will_resolve"
    is_dynamic = True

    def compute_label(self, ctx: TrajectoryContext) -> list[bool | None]:
        return [ctx.sample["outcome"]] * ctx.n_captured_steps
