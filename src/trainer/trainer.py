"""
Generic trainer class template.
"""

import os
from tqdm import tqdm
from typing import Optional
from datetime import datetime

import torch
import torch.amp as amp
from torch.amp.grad_scaler import GradScaler
from torchinfo import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from src.logging.logger import get_logger
from src.utils import save_audio
from .checkpointer import Checkpointer
from .factory import (
    get_criterions,
    get_metrics,
    get_optimizers,
    get_schedulers,
)


logger = get_logger(__name__)


class ModelTrainer:
    def __init__(
        self, config: dict, model_dir: str, model: torch.nn.Module, device: torch.device
    ):
        self.config = config
        self.model_dir = model_dir
        assert os.path.exists(self.model_dir), (
            f"Model directory {self.model_dir} does not exist."
        )
        self.device = device
        self.model = model
        self.model.to(self.device)

        self.clip_norm = self.config["trainer"]["clip_norm"]

        assert len(self.config["criterions"]) != 0
        assert len(self.config["optimizers"]) != 0

        self.criterions = get_criterions(self.config["criterions"])
        self.criterions_stats = {
            c["name"]: {"ema": 1.0, "weight": c.get("weight", 1.0)}
            for c in self.config["criterions"]
        }

        self.optimizers = get_optimizers(self.config["optimizers"], model.parameters())
        self.schedulers = get_schedulers(self.config["schedulers"], self.optimizers)
        self.start_scheduling = config["trainer"]["start_scheduling"]

        self.scaler = None
        if self.config["trainer"]["precision"] == "fp16" and self.device.type == "cuda":
            logger.info("Using mixed precision & GradScaler")
            self.scaler = GradScaler()

        self.metrics = get_metrics(self.config["metrics"])

        self.checkpoints_dir = os.path.join(self.model_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.checkpointer = Checkpointer(
            self.config,
            self.model,
            self.optimizers,
            self.schedulers,
            self.scaler,
            self.checkpoints_dir,
        )

        logger.info("=== Model summary ===")
        with open(os.path.join(self.model_dir, "model_summary.txt"), "w") as f:
            f.write(str(summary(self.model, device=device)))

    def log_tensorboard_metrics(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ):
        logger.info("Logging metrics to tensorboard")
        for k in train_metrics.keys() | val_metrics.keys():
            self.tensorboard_writer.add_scalars(
                k,
                {
                    "Train": train_metrics.get(k, None),
                    "Val": val_metrics.get(k, None),
                },
                epoch,
            )

        for i, optimizer in enumerate(self.optimizers):
            for j, param_group in enumerate(optimizer.param_groups):
                self.tensorboard_writer.add_scalars(
                    "LR",
                    {f"optimizer_{i}_group_{j}": param_group["lr"]},
                    epoch,
                )
        self.tensorboard_writer.flush()

    def update_schedulers(
        self,
        step: bool = False,
        epoch: bool = False,
        val_loss: Optional[float] = None,
    ):
        """
        Update learning rate schedulers.

        Args:
        - step: If True, update step-based schedulers (called per batch).
        - epoch: If True, update epoch-based schedulers (called per epoch).
        - val_loss: (Optional) Validation loss for ReduceLROnPlateau.
        """
        logger.info("Updating schedulers")
        for scheduler in self.schedulers:
            if epoch and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if val_loss is not None:
                    scheduler.step(val_loss)
                else:
                    logger.warning(
                        "Couldn't update ReduceLROnPlateau scheduler because val_loss is None"
                    )

            elif step and isinstance(
                scheduler,
                (
                    torch.optim.lr_scheduler.OneCycleLR,
                    torch.optim.lr_scheduler.LambdaLR,
                    torch.optim.lr_scheduler.ExponentialLR,
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
                ),
            ):
                scheduler.step()  # update step-based schedulers

            elif epoch and isinstance(
                scheduler,
                (
                    torch.optim.lr_scheduler.StepLR,
                    torch.optim.lr_scheduler.MultiStepLR,
                    torch.optim.lr_scheduler.CosineAnnealingLR,
                ),
            ):
                scheduler.step()  # update epoch-based schedulers

    def update_metrics(self, logits: torch.Tensor, targets: torch.Tensor):
        """Updates each evaluation metric"""
        for metric in self.metrics.values():
            metric.update(
                # reshape to [batch_size * channels, time]
                logits.view(logits.shape[0], -1).cpu().detach(),
                targets.view(targets.shape[0], -1).cpu().detach(),
            )

    def compute_metrics(self) -> dict:
        """
        Computes each evaluation metric.
        Evaluation metrics are updated per batch in both training and validation.

        Returns:
        - Dictionary containing names of metrics defined in the configuration and their corresponding results
        """
        logger.info("Computing metrics")
        results = {}
        for m, f in self.metrics.items():
            results[m] = f.compute()
            f.reset()

        return results

    def train_one_epoch(
        self, data_loader: DataLoader, update_schedulers: bool = False
    ) -> dict:
        """
        Args:
            data_loader: DataLoader for training data.
        Returns:
        - Epoch training metrics. Check compute_metrics() for more details.
        """
        self.model.train()
        running_loss = 0.0

        for _, eid, inputs, targets in tqdm(
            data_loader, total=len(data_loader), desc="Training"
        ):
            eid, inputs, targets = (
                eid.to(self.device),
                inputs.to(self.device),
                targets.to(self.device),
            )
            inputs = (eid, inputs)

            # zero out gradients for all optimizers
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            with amp.autocast(
                device_type=self.device.type, enabled=self.scaler is not None
            ):
                logits = self.model(inputs)
                # sum and normalize losses

                loss = 0
                raw_losses = {}
                for name, criterion in self.criterions.items():
                    raw_loss = criterion(logits, targets)
                    raw_losses[name] = raw_loss.detach()  # keep raw loss for EMA update
                    ema = self.criterions_stats[name]["ema"]
                    normalized_loss = raw_loss / (ema + 1e-8)
                    loss += normalized_loss * self.criterions_stats[name]["weight"]

            # backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                for optimizer in self.optimizers:
                    if self.clip_norm:
                        self.scaler.unscale_(optimizer)
                        clip_grad_norm_(
                            self.model.parameters(), max_norm=self.clip_norm
                        )

                    self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                for optimizer in self.optimizers:
                    if self.clip_norm:
                        clip_grad_norm_(
                            self.model.parameters(), max_norm=self.clip_norm
                        )
                    optimizer.step()

            # accumulate loss
            running_loss += loss.item()

            # update schedulers for STEP only
            if update_schedulers:
                self.update_schedulers(step=True, epoch=False, val_loss=None)

            for name, raw_loss in raw_losses.items():
                self.criterions_stats[name]["ema"] = (
                    0.99 * self.criterions_stats[name]["ema"] + 0.01 * raw_loss.item()
                )
            # update the metrics
            self.update_metrics(logits, targets)

        # normalize loss by dataset size
        avg_loss = running_loss / len(data_loader.dataset)  # pyright: ignore

        return {"Loss": avg_loss, **self.compute_metrics()}

    def train(self, max_epochs: int, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train the model over multiple epochs. The last checkpoint is automatically loaded.
        Args:
            max_epochs: maximum number of epochs to train the model for.
            train_loader: training set DataLoader.
            val_loader: validation set DataLoader.
        """
        # create logs dir and tensorboard loggers for current training run
        self.logs_dir = os.path.join(
            self.model_dir, "logs", datetime.now().strftime("%d%m%Y-%H%M%S")
        )
        os.makedirs(self.logs_dir, exist_ok=True)

        self.tensorboard_writer = SummaryWriter(log_dir=self.logs_dir)

        # load last checkpoint (or start from scratch if 1)
        start_epoch = self.checkpointer.load_checkpoint(self.device)
        # log training info
        logger.info(
            f"Starting training from epoch: {start_epoch} for {max_epochs} max epochs."
        )

        train_metrics = {}
        val_metrics = {}
        try:
            for epoch in range(start_epoch, max_epochs):
                logger.info(f"Entering Epoch: {epoch}/{max_epochs}")
                train_metrics = self.train_one_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)

                if epoch >= self.start_scheduling:
                    self.update_schedulers(
                        step=False,
                        epoch=True,
                        val_loss=val_metrics["Loss"],
                    )
                else:
                    logger.info(
                        f"Current epoch {epoch} < {self.start_scheduling}, skipping scheduling"
                    )

                self.log_tensorboard_metrics(epoch, train_metrics, val_metrics)
                self.checkpointer.save_checkpoint(epoch, val_loss=val_metrics["Loss"])

                logger.info("=== Epoch results ===")
                for k in train_metrics.keys() | val_metrics.keys():
                    logger.info(
                        f"{k}: Train={train_metrics.get(k, 'N/A'):.4f} | Val={val_metrics.get(k, 'N/A'):.4f}"
                    )

        except KeyboardInterrupt:
            logger.warning("Training interrupted. Saving checkpoint and exiting")
            self.checkpointer.save_checkpoint(epoch)

        finally:
            logger.info("Finished training")
            self.close_writer()

    def evaluate(
        self, data_loader: DataLoader, output_dir: Optional[str] = None
    ) -> dict:
        """
        Args:
            data_loader: validation set DataLoader.
        Returns:
        - Evaluation metrics. Check compute_metrics() for more details.
        """
        logger.info("Evaluating model")
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0

            try:
                for id, eid, inputs, targets in tqdm(
                    data_loader, total=len(data_loader), desc="Evaluation"
                ):
                    eid, inputs, targets = (
                        eid.to(self.device),
                        inputs.to(self.device),
                        targets.to(self.device),
                    )
                    inputs = (eid, inputs)
                    logits = self.model(inputs)

                    if output_dir is not None:
                        save_audio(logits, data_loader.dataset.fs, id, output_dir)

                    # sum and normalize losses
                    losses = [
                        (
                            criterion(logits, targets)
                            * self.criterions_stats[name]["weight"]
                            / (self.criterions_stats[name]["ema"] + 1e-8)
                        )
                        for name, criterion in self.criterions.items()
                    ]
                    running_loss += sum(losses).item() * targets.shape[0]

                    self.update_metrics(logits, targets)
            except KeyboardInterrupt:
                logger.warning("Evaluation interrupted. Computing current results")

            # normalize loss by dataset size
            avg_loss = running_loss / len(data_loader.dataset)

            return {"Loss": avg_loss, **self.compute_metrics()}

    def test(self, test_loader, output_dir: Optional[str]):
        """
        Evaluate the model on the test data
        Args:
            test_loader: test DataLoader.
        """
        logger.info("Testing model")
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving output to: {output_dir}")

        metrics = self.evaluate(test_loader)
        logger.info("=== Test results ===")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")

    def close_writer(self):
        """
        Close the Tensorboard writer.
        """
        logger.info("Closing tensorboard writer")
        self.tensorboard_writer.close()
