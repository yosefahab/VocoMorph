import os
import torch
import glob
from typing import Optional
from src.logging.logger import get_logger

logger = get_logger(__name__)


class Checkpointer:
    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        optimizers,
        schedulers,
        scaler,
        checkpoints_dir: str,
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

        self.config = config
        self.checkpoint_path = config.get("checkpoint_path", None)
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.scaler = scaler
        self.checkpoint_dir = checkpoints_dir
        self.save_interval = config.get("save_interval", 1)
        self.keep_last_n = config.get("keep_last_n", None)
        self.save_best = config.get("save_best", None)
        self.best_val_loss = float("inf")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        """
        Saves a model checkpoint with different strategies.

        Args:
            epoch: The current epoch number.
            val_loss: Validation loss (used if save_best=True).
        """
        if self.save_best and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, "best_checkpoint.pt"
                )
                self._save_to_disk(checkpoint_path, epoch)
                logger.info(f"New best model saved (val_loss={val_loss:.4f})")
                return

        if epoch % self.save_interval == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"ckpt_epoch_{epoch}.pt"
            )
            logger.info(
                f"New checkpoint saved per save_interval: ({self.save_interval})"
            )
            self._save_to_disk(checkpoint_path, epoch)

            # remove old checkpoints if keep_last_n is set
            if self.keep_last_n:
                self._cleanup_old_checkpoints()

    def _save_to_disk(self, checkpoint_path: str, epoch: int):
        """
        Saves model and optimizer states to disk.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dicts": [opt.state_dict() for opt in self.optimizers],
                "scheduler_state_dicts": [sch.state_dict() for sch in self.schedulers],
                "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                "epoch": epoch,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _cleanup_old_checkpoints(self):
        """
        Removes old checkpoints if the number of saved checkpoints exceeds keep_last_n.
        """
        checkpoint_files = sorted(
            glob.glob(os.path.join(self.checkpoint_dir, "ckpt_epoch_*.pt")),
            key=lambda f: int(os.path.splitext(f.split("_")[-1])[0]),
        )

        if self.keep_last_n and len(checkpoint_files) > self.keep_last_n:
            logger.info("Cleaning up old checkpoints per keep_last_n")
            to_remove = checkpoint_files[: len(checkpoint_files) - self.keep_last_n]
            for old_checkpoint in to_remove:
                os.remove(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, device: torch.device) -> int:
        """
        Loads a checkpoint (latest if no path is specified).

        Args:
            checkpoint_path: Path to a specific checkpoint (optional).
        Returns:
            The epoch number to resume from.
        """
        if self.checkpoint_path is None:
            self.checkpoint_path = self._get_latest_checkpoint()
            if self.checkpoint_path is None:
                logger.info("No checkpoints found, starting from scratch.")
                return 1

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(
            self.checkpoint_path, map_location=device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dicts" in checkpoint:
            for opt, state in zip(self.optimizers, checkpoint["optimizer_state_dicts"]):
                opt.load_state_dict(state)

        if "scheduler_state_dicts" in checkpoint:
            for sch, state in zip(self.schedulers, checkpoint["scheduler_state_dicts"]):
                sch.load_state_dict(state)

        if (
            self.scaler
            and "scaler_state_dict" in checkpoint
            and checkpoint["scaler_state_dict"] is not None
        ):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint.get("epoch", 1)

    def _get_latest_checkpoint(self) -> Optional[str]:
        """
        Finds the latest checkpoint in the directory.
        Returns:
            The path to the latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = [
            f
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("ckpt_epoch_") and f.endswith(".pt")
        ]
        if not checkpoints:
            return None

        checkpoints.sort(key=self._get_epoch_from_name)
        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    @staticmethod
    def _get_epoch_from_name(filename: str) -> int:
        """Extracts epoch number from filename"""
        return int(os.path.splitext(filename)[0].split("_")[-1])
