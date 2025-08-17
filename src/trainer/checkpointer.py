from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.utils import tree_flatten

from src.utils.logger import get_logger
from src.utils.types import DictConfig


class Checkpointer:
    _MODEL_STATE = "model_state_dict"
    _EPOCH = "epoch"

    def __init__(self, checkpoints_dir: Path, config: DictConfig, model: nn.Module):
        """
        Handles saving and loading checkpoints for MLX models.

        Args:
            checkpoints_dir: Path to save checkpoints.
            config: Dictionary with configuration keys (e.g., save_interval, save_best, keep_last_n).
            model: The MLX model to save/load.
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
        self.checkpoint_dir = checkpoints_dir
        self.save_interval = config.get("save_interval", 1)
        self.keep_last_n = config.get("keep_last_n", None)
        self.save_best = config.get("save_best", False)
        self.best_val_loss = float("inf")

        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.logger.info(f"Using save interval: {self.save_interval} for checkpointer")
        if self.keep_last_n:
            self.logger.info(
                f"Using keep last {self.keep_last_n} policy for checkpointer"
            )
        if self.save_best:
            self.logger.info("Using save best policy for checkpointer")

    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        """
        Saves a model checkpoint based on strategy configuration.

        Args:
            epoch: Current training epoch.
            val_loss: Optional validation loss for best-checkpoint tracking.
        """
        if self.save_best and val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = self.checkpoint_dir.joinpath("best_checkpoint.npz")
            self._save_to_disk(checkpoint_path, epoch)
            self.logger.info(f"New best model saved per best (val_loss={val_loss:.4f})")

        if epoch % self.save_interval == 0:
            checkpoint_path = self.checkpoint_dir.joinpath(f"ckpt_epoch_{epoch}.npz")
            self._save_to_disk(checkpoint_path, epoch)
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
        mx.savez(
            file=str(checkpoint_path),
            **dict(tree_flatten(self.model.trainable_parameters())),
        )
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _cleanup_old_checkpoints(self):
        """
        Removes older checkpoints if total exceeds `keep_last_n`.
        """
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("ckpt_epoch_*.npz")],
            key=lambda f: int(f.stem.split("_")[-1]),
        )
        if self.keep_last_n is not None and len(checkpoints) > self.keep_last_n:
            to_remove = checkpoints[: len(checkpoints) - self.keep_last_n]
            for old_checkpoint in to_remove:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(
        self, prioritize_best: bool = True, custom_path: Optional[Path] = None
    ) -> int:
        """
        Loads a checkpoint into the model using the path specified in the config.

        Args:
            prioritize_best: Prefer 'best_checkpoint.npz' if it exists.
            custom_path: Optional custom path to a specific checkpoint.

        Returns:
            Epoch number to resume from (epoch + 1).
        """
        checkpoint_path = custom_path or self.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint(prioritize_best)
            if checkpoint_path is None:
                self.logger.warning("No checkpoints found, starting from scratch.")
                return 1

        if not checkpoint_path.exists():
            self.logger.warning(
                "No checkpoints found at specified path, starting from scratch."
            )
            return 1

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            self.model.load_weights(str(checkpoint_path))  # pyright: ignore[reportOptionalCall]
            return 1

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
            return 1

    def _get_latest_checkpoint(self, prioritize_best: bool = True) -> Optional[Path]:
        """
        Returns latest checkpoint path or best if found first.
        """
        if prioritize_best:
            best_ckpt = self.checkpoint_dir.joinpath("best_checkpoint.npz")
            if best_ckpt.exists():
                return best_ckpt

        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob("ckpt_epoch_*.npz")],
            key=lambda f: int(f.stem.split("_")[-1]),
        )
        return checkpoints[-1] if checkpoints else None


class NullCheckpointer(Checkpointer):
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        pass
