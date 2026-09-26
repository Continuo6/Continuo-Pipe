"""Control-flow exceptions used by the pipeline."""


class PipelineError(Exception):
    """Base class for pipeline-level errors."""


class PipelineAbort(PipelineError):
    """Raised by a stage to signal that the current audio should be skipped.

    The orchestrator catches this, writes an empty result JSON, and moves on
    to the next file. Equivalent to the early-return branches in the
    original ``main_process``.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class StageConfigError(PipelineError):
    """Raised when a stage is built with an invalid configuration."""
