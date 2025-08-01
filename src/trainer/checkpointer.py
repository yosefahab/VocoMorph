from pathlib import Path
from typing import List, Optional

import torch

from src.utils.logger import get_logger
from src.utils.types import DictConfig


class Checkpointer:
    _MODEL_STATE = "model_state_dict"
    _OPTIMIZER_STATES = "optimizers_states_dicts"
    _SCHEDULER_STATES = "schedulers_states_dicts"
    _SCALER_STATE = "scaler_state_dict"
    _EPOCH = "epoch"

    def __init__(
        self,
        checkpoints_dir: Path,
        config: DictConfig,
        model: torch.nn.Module | torch.nn.DataParallel,
        optimizers: List[torch.optim.Optimizer],
        schedulers: List[Optional[torch.optim.lr_scheduler._LRScheduler]] = [],
        scaler: Optional[torch.GradScaler] = None,
    ):
        """
        Handles saving and loading checkpoints with configurable strategies.

        Args:
            config: Dictionary with configuration keys (e.g., save_interval, save_best, keep_last_n).
            model: Model to save/load.
            optimizers: List of optimizers.
            schedulers: List of schedulers (can contain None).
            scaler: AMP gradient scaler.
            checkpoints_dir: Path to save checkpoints.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.config = config
        checkpoint_path = config.get("checkpoint_path")
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

        if self.checkpoint_path and not self.checkpoint_path.exists():
            self.logger.warning(
                f"Found checkpoint path in config file that doesn't exist: {self.checkpoint_path}"
            )

        self.model = model
        self.optimizers = optimizers or []
        self.schedulers = schedulers or []
        self.scaler = scaler
        self.checkpoint_dir = checkpoints_dir

        self.save_interval = config.get("save_interval", 1)
        self.logger.info(f"Using save interval: {self.save_interval} for checkpointer")

        self.keep_last_n = config.get("keep_last_n", None)
        if self.keep_last_n:
            self.logger.info(
                f"Using keep last {self.keep_last_n} policy for checkpointer"
            )

        self.save_best = config.get("save_best", False)
        if self.save_best:
            self.logger.info("Using save best policy for checkpointer")

        self.best_val_loss = float("inf")

        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        """
        Saves a model checkpoint based on strategy configuration.

        Args:
            epoch: Current training epoch.
            val_loss: Optional validation loss for best-checkpoint tracking.
        """
        if self.save_best and val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = self.checkpoint_dir.joinpath("best_checkpoint.pt")
            self._save_to_disk(checkpoint_path, epoch)
            self.logger.info(f"New best model saved per best (val_loss={val_loss:.4f})")

        checkpoint_path = self.checkpoint_dir.joinpath(f"ckpt_epoch_{epoch}.pt")
        self._save_to_disk(checkpoint_path, epoch)

        if epoch % self.save_interval == 0:
            self.logger.info(
                f"New checkpoint saved per save_interval: {self.save_interval}"
            )
            if self.keep_last_n:
                self._cleanup_old_checkpoints()

    def _save_to_disk(self, checkpoint_path: Path, epoch: int):
        """
        Internal: Save model and training states to disk.

        Args:
            checkpoint_path: Filepath to save to.
            epoch: Current epoch.
        """
        optimizers_states_dicts = [opt.state_dict() for opt in self.optimizers]
        schedulers_states_dicts = [
            sch.state_dict() if sch is not None else None for sch in self.schedulers
        ]
        scaler_state_dict = self.scaler.state_dict() if self.scaler else None

        torch.save(
            {
                self._MODEL_STATE: self.model.state_dict(),
                self._OPTIMIZER_STATES: optimizers_states_dicts,
                self._SCHEDULER_STATES: schedulers_states_dicts,
                self._SCALER_STATE: scaler_state_dict,
                self._EPOCH: epoch,
            },
            checkpoint_path,
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _cleanup_old_checkpoints(self):
        """
        Removes older checkpoints if total exceeds `keep_last_n`.
        """
        checkpoints = list(self.checkpoint_dir.glob("ckpt_epoch_*.pt"))
        checkpoint_files = sorted(checkpoints, key=lambda f: int(f.stem.split("_")[-1]))

        assert isinstance(self.keep_last_n, int)

        if len(checkpoint_files) <= self.keep_last_n:
            return

        self.logger.info("Cleaning up old checkpoints per keep_last_n")
        to_remove = checkpoint_files[: len(checkpoint_files) - self.keep_last_n]
        for old_checkpoint in to_remove:
            old_checkpoint.unlink()
            self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(
        self,
        device: torch.device,
        prioritize_best: bool = True,
        custom_path: Optional[Path] = None,
    ) -> int:
        """
        Loads a checkpoint into model and training components using the path specified in the model config file.
        Args:
        - device: Device to map loaded state.
        Returns:
            Epoch number to resume from (epoch + 1).
        """
        checkpoint_path = custom_path or self.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path = self._get_latest_checkpoint(
                prioritize_best
            )
            if checkpoint_path is None:
                self.logger.warning("No checkpoints found, starting from scratch.")
                return 1
        else:
            assert checkpoint_path.exists()

        self.logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        self.model.load_state_dict(checkpoint[self._MODEL_STATE])

        if self._OPTIMIZER_STATES not in checkpoint:
            self.logger.error(
                f"{self._OPTIMIZER_STATES} missing from checkpoint state dict"
            )
        else:
            print(len(checkpoint[self._OPTIMIZER_STATES]), len(self.optimizers))
            for opt, state in zip(
                self.optimizers, checkpoint[self._OPTIMIZER_STATES], strict=True
            ):
                opt.load_state_dict(state)

        if self._SCHEDULER_STATES not in checkpoint:
            self.logger.warning(
                f"{self._SCHEDULER_STATES} missing from checkpoint state dict"
            )
        else:
            for sch, state in zip(
                self.schedulers, checkpoint[self._SCHEDULER_STATES], strict=True
            ):
                if sch is not None:
                    sch.load_state_dict(state)

        if self.scaler is not None:
            if self._SCALER_STATE not in checkpoint:
                self.logger.warning(
                    "scaler_state_dict missing from checkpoint state dict"
                )
            elif checkpoint[self._SCALER_STATE] is not None:
                self.scaler.load_state_dict(checkpoint[self._SCALER_STATE])

        return checkpoint.get("epoch", 0) + 1

    def _get_latest_checkpoint(self, prioritize_best: bool = True) -> Optional[Path]:
        """
        Returns latest checkpoint path or best if found first.

        Args:
            prioritize_best: Prefer 'best_checkpoint.pt' if it exists.

        Returns:
            Path to latest or best checkpoint, or None if none exist.
        """
        if prioritize_best:
            best_ckpt = self.checkpoint_dir.joinpath("best_checkpoint.pt")
            if best_ckpt.exists():
                return best_ckpt

        checkpoints = list(self.checkpoint_dir.glob("ckpt_epoch_*.pt"))
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda f: int(f.stem.split("_")[-1]))
        return checkpoints[-1]


class NullCheckpointer(Checkpointer):
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        pass
