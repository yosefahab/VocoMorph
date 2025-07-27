from pathlib import Path
from typing import List, Optional

import torch

from src.utils.logger import get_logger


class Checkpointer:
    _MODEL_STATE = "model_state_dict"
    _OPTIMIZER_STATES = "optimizers_states_dicts"
    _SCHEDULER_STATES = "schedulers_states_dicts"
    _SCALER_STATE = "scaler_state_dict"
    _EPOCH = "epoch"

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module | torch.nn.DataParallel,
        optimizers: Optional[List[torch.optim.Optimizer]],
        schedulers: Optional[List[torch.optim.lr_scheduler._LRScheduler]],
        scaler: Optional[torch.GradScaler],
        checkpoints_dir: Path,
        is_distributed: bool = False,
    ):
        """
        Handles saving and loading checkpoints with configurable strategies.
        Args:
        - model: The model to save/load.
        - optimizers: List of optimizers.
        - schedulers: List of schedulers.
        - scaler: Gradient scaler (if using mixed precision).
        - checkpoint_dir: Directory where checkpoints are stored.
        """
        self.logger = get_logger(self.__class__.__name__)

        self.is_distributed = is_distributed
        self.config = config
        self.checkpoint_path = config.get("checkpoint_path", None)
        if self.checkpoint_path:
            self.logger.info(f"Starting from checkpoint: {self.checkpoint_path}")

        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
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
        Saves a model checkpoint with different strategies.
        Args:
        - epoch: The current epoch number.
        - val_loss: Validation loss (used if checkpointer is configured with save_best=True).
        """
        if self.is_distributed and torch.distributed.get_rank() != 0:
            return
        if self.save_best and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = self.checkpoint_dir.joinpath("best_checkpoint.pt")
                self._save_to_disk(checkpoint_path, epoch)
                self.logger.info(
                    f"New best model saved per best (val_loss={val_loss:.4f})"
                )
                return

        checkpoint_path = self.checkpoint_dir.joinpath(f"ckpt_epoch_{epoch}.pt")
        self._save_to_disk(checkpoint_path, epoch)
        if epoch % self.save_interval == 0:
            self.logger.info(
                f"New checkpoint saved per save_interval: {self.save_interval}"
            )

            # remove old checkpoints if keep_last_n is set
            if self.keep_last_n:
                self._cleanup_old_checkpoints()

    def _save_to_disk(self, checkpoint_path: Path, epoch: int):
        """
        Saves model and optimizer states to disk as well as schedulers, grad scalers if they exist.
        """
        optimizers_states_dicts = (
            [opt.state_dict() for opt in self.optimizers] if self.optimizers else None
        )
        schedulers_states_dicts = (
            [sch.state_dict() for sch in self.schedulers] if self.schedulers else None
        )
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
        Removes old checkpoints if the number of saved checkpoints exceeds keep_last_n.
        """

        checkpoints = list(self.checkpoint_dir.glob("ckpt_epoch_*.pt"))
        checkpoint_files = sorted(
            checkpoints,
            key=lambda f: int(f.stem.split("_")[-1]),
        )

        assert isinstance(self.keep_last_n, int), (
            "keep_last_n must be an int to use this feature"
        )

        if len(checkpoint_files) <= self.keep_last_n:
            return

        self.logger.info("Cleaning up old checkpoints per keep_last_n")
        to_remove = checkpoint_files[: len(checkpoint_files) - self.keep_last_n]
        for old_checkpoint in to_remove:
            old_checkpoint.unlink()
            self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, device: torch.device) -> int:
        """
        Loads a checkpoint (latest if no path is specified).
        Args:
        - checkpoint_path: Path to a specific checkpoint (optional).
        Returns:
            The epoch number to resume from.
        """
        if self.checkpoint_path is None:
            self.checkpoint_path = self._get_latest_checkpoint()
            if self.checkpoint_path is None:
                self.logger.info("No checkpoints found, starting from scratch.")
                return 1

        self.logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path, map_location=device, weights_only=False
        )
        self.model.load_state_dict(checkpoint[self._MODEL_STATE])

        if self.optimizers:
            if self._OPTIMIZER_STATES not in checkpoint:
                self.logger.error(
                    f"{self._OPTIMIZER_STATES} missing from checkpoint state dict"
                )
            else:
                for opt, state in zip(
                    self.optimizers, checkpoint[self._OPTIMIZER_STATES]
                ):
                    opt.load_state_dict(state)

        if self.schedulers is not None:
            if self._SCHEDULER_STATES not in checkpoint:
                self.logger.error(
                    f"{self._SCHEDULER_STATES} missing from checkpoint state dict"
                )
            else:
                for sch, state in zip(
                    self.schedulers, checkpoint[self._SCHEDULER_STATES]
                ):
                    sch.load_state_dict(state)

        if self.scaler is not None:
            if self._SCALER_STATE not in checkpoint:
                self.logger.error(
                    "scaler_state_dict missing from checkpoint state dict"
                )
            elif checkpoint[self._SCALER_STATE] is not None:
                self.scaler.load_state_dict(checkpoint[self._SCALER_STATE])

        return checkpoint.get("epoch", 0) + 1

    def _get_latest_checkpoint(self, prioritize_best: bool = True) -> Optional[Path]:
        """
        Finds the latest checkpoint in the directory.
        Args:
        - prioritize_best: If True, return 'best_checkpoint.pt' if it exists.
        Returns:
            Path to the latest checkpoint, or None if none found.
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
