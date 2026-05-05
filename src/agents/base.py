from abc import ABC, abstractmethod


class AgentAdapter(ABC):
    """
    Wraps a ModelAdapter and TaskAdapter into an agentic episode loop that
    produces a TrajectoryContext with a populated edit_history. Intended for
    scaffolds such as SWE-agent, MiniSWEAgent, nano-agent, etc.

    Not implemented in this version. To add a scaffold:
      1. Subclass AgentAdapter and implement run_episode().
      2. Pass it to run_extract.py via --agent-config (future flag).
      3. Dynamic probes (currently_correct, etc.) become available automatically,
         since they only require a non-empty edit_history.

    In non-agent mode, run_extract.py calls model_adapter.generate() directly
    and edit_history remains empty, making dynamic probes unavailable.
    """

    @abstractmethod
    def run_episode(
        self, sample: dict, model_adapter, task_adapter,
    ) -> "TrajectoryContext": ...
