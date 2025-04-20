from dataclasses import dataclass, field, fields
from typing import Callable
from functools import partial
import torch


@dataclass(slots=False)
class WarmupConstantSchedule(torch.optim.lr_scheduler.LambdaLR):
    optimizer: torch.optim.Optimizer
    warmup_steps: int
    last_epoch: int = field(default=-1)
    lr_lambda: Callable[[int], float] = field(init=False)

    @staticmethod
    def _lr_lambda(step: int, warmup_steps: int) -> float:
        """Learning rate schedule with linear warmup to constant value (static method)."""
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0

    def __post_init__(self):
        self.lr_lambda = partial(self._lr_lambda, warmup_steps=self.warmup_steps)
        super().__init__(self.optimizer, self.lr_lambda, last_epoch=self.last_epoch)

    def __repr__(self):
        class_name = self.__class__.__name__
        init_fields = [f for f in fields(self) if f.init]
        field_strs = [
            f"{field.name}={getattr(self, field.name)!r}" for field in init_fields
        ]
        lr_lambda_repr = f"lr_lambda = partial(<function WarmupConstantSchedule._lr_lambda at ...>, warmup_steps={self.warmup_steps})"
        return f"<{class_name}({', '.join(field_strs + [lr_lambda_repr])})>"
