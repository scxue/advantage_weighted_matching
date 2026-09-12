"""One W&B history row per training step, with replay-safe resume logging."""
from numbers import Integral


class TrainingStepLogger:
    def __init__(self, log, *, next_step=0):
        # Read run.step once after wandb.init: W&B, not the model checkpoint,
        # knows which history rows were already received before interruption.
        self._next_step = self._validate_step(next_step)
        self._log = log
        self._pending_step = None

    @staticmethod
    def _validate_step(step):
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("W&B training step must be a nonnegative integer")
        return int(step)

    def log(self, data, *, step):
        step = self._validate_step(step)
        if step < self._next_step:
            # Replayed training/eval must still execute, but never re-upload
            # its metrics or media into an already committed history step.
            return False
        if self._pending_step is not None:
            if step < self._pending_step:
                raise ValueError("Cannot log backwards within an uncommitted training step")
            if step > self._pending_step:
                # Also support configurations with several updates per epoch.
                self.flush()
        # Serialize media now, while callers' TemporaryDirectories still exist.
        # commit=False merges this fragment without advancing W&B's history step.
        self._log({**data, "train/global_step": step}, step=step, commit=False)
        self._pending_step = step
        return True

    def flush(self):
        if self._pending_step is None:
            return
        step = self._pending_step
        self._log({}, step=step, commit=True)
        self._next_step = step + 1
        self._pending_step = None
