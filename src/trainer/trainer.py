import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch

# from ptflops import get_model_complexity_info
# from thop import profile
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from src.utils.audio import save_audio
from src.utils.logger import get_logger

from .checkpointer import Checkpointer
from .factory import get_criterions, get_metrics, get_optimizers_and_schedulers


class ModelTrainer:
    def __init__(
        self,
        config: dict,
        model_dir: Path,
        model: torch.nn.Module,
        device: torch.device,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        assert model_dir.exists(), f"Model directory {model_dir} does not exist."
        self.model_dir = model_dir
        self.device = device
        self.model = model
        self.model.to(self.device)

        self.clip_norm = self.config["trainer"]["clip_norm"]

        self.criterions = get_criterions(self.config["criterions"])
        self.criterions_stats = {
            c["name"]: {"weight": c.get("weight", 1.0)}
            for c in self.config["criterions"]
        }

        self.optimizers_and_schedulers = get_optimizers_and_schedulers(
            self.config["optimizers"], model.parameters()
        )

        self.optimizers = []
        self.schedulers = []
        for item in self.optimizers_and_schedulers:
            self.optimizers.append(item["optimizer"])
            if item["scheduler"] is not None:
                self.schedulers.append(item["scheduler"])

        self.start_scheduling = config["trainer"]["start_scheduling"]
        self.grad_accum_steps = self.config["trainer"].get("grad_accumulation_steps", 1)
        self.logger.info(f"Using gradient accumulation steps: {self.grad_accum_steps}")

        self.scaler = None
        if self.config["trainer"]["precision"] == "fp16" and self.device.type == "cuda":
            self.logger.info("Using mixed precision & GradScaler")
            self.scaler = GradScaler()

        self.test_epochs = self.config["trainer"].get("test_epochs", [])
        self.metrics = get_metrics(self.config["metrics"])

        self.checkpoints_dir = self.model_dir.joinpath("checkpoints")
        self.checkpoints_dir.mkdir(exist_ok=True)

        self.checkpointer = Checkpointer(
            self.config["checkpointer"],
            self.model,
            self.optimizers,
            self.schedulers,
            self.scaler,
            self.checkpoints_dir,
        )
        self._get_model_summary()

    def _get_model_summary(self):
        self.logger.info("Generating model summary")
        dummy_input_shape = self.config["trainer"].get("dummy_input")

        col_names = ["num_params", "params_percent", "kernel_size", "trainable"]
        dummy_input = None
        if dummy_input_shape:
            dummy_input = self._generate_dummy_input(dummy_input_shape)
            self.logger.info(f"Summary dummy input: {dummy_input}")
            if dummy_input is not None:
                col_names.extend(["input_size", "output_size", "mult_adds"])
                dummy_input = (dummy_input,)

        model_summary = summary(
            self.model,
            input_data=dummy_input,
            device=self.device,
            verbose=0,
        )

        summary_file_path = self.model_dir.joinpath("model_summary.txt")
        with open(summary_file_path, "w") as f:
            self.logger.info(f"Saving model summary to: {summary_file_path}")
            f.write(str(model_summary))

    def _generate_dummy_input(self, dummy_cfg):
        self.logger.info(f"Generating dummy input from: {dummy_cfg}")

        def resolve_dtype(dtype_str):
            try:
                return getattr(torch, dtype_str)
            except AttributeError:
                self.logger.warning(
                    f"Invalid dtype: {dtype_str}, defaulting to float32"
                )
                return torch.float32

        def make_tensor(entry):
            if isinstance(entry, dict) and "shape" in entry:
                shape = entry["shape"]
                dtype = resolve_dtype(entry.get("dtype", "float32"))
                if dtype.is_floating_point:
                    return torch.rand(*shape, dtype=dtype)
                else:
                    return torch.zeros(*shape, dtype=dtype)
            elif isinstance(entry, list):
                return torch.rand(*entry, dtype=torch.float32)
            elif isinstance(entry, (int, float)):
                return torch.tensor(entry, dtype=torch.float32)
            else:
                self.logger.error(f"Unsupported dummy_input entry: {entry}")
                return None

        if isinstance(dummy_cfg, list):
            tensors = tuple(make_tensor(item) for item in dummy_cfg)
            return tensors if len(tensors) > 1 else tensors[0]
        elif isinstance(dummy_cfg, dict):
            return {k: make_tensor(v) for k, v in dummy_cfg.items()}
        else:
            return make_tensor(dummy_cfg)

    def _log_tensorboard_metrics(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
    ):
        self.logger.info("Logging metrics to tensorboard")

        self.logger.info("=== Epoch losses ===")
        train_loss_keys = set(train_metrics["Losses"].keys())
        val_loss_keys = set(val_metrics["Losses"].keys())
        diff = train_loss_keys ^ val_loss_keys
        assert not diff, f"Discrepancy between train and val losses: {diff}"

        for metric in train_loss_keys:
            t_metric = train_metrics["Losses"][metric]
            v_metric = val_metrics["Losses"][metric]
            self.logger.info(f"{metric}: Train={t_metric:.4f} | Val={v_metric:.4f}")

            self.tensorboard_writer.add_scalars(
                f"Losses/{metric}",
                {"Train": t_metric, "Val": v_metric},
                epoch,
            )

        self.logger.info("=== Epoch metrics ===")
        train_metric_keys = set(train_metrics["Metrics"].keys())
        val_metric_keys = set(val_metrics["Metrics"].keys())
        diff = train_metric_keys ^ val_metric_keys
        assert not diff, f"Discrepancy between train and val metrics: {diff}"

        for metric in train_metric_keys:
            t_metric = train_metrics["Metrics"][metric]
            v_metric = val_metrics["Metrics"][metric]
            self.logger.info(f"{metric}: Train={t_metric:.4f} | Val={v_metric:.4f}")

            self.tensorboard_writer.add_scalars(
                f"Metrics/{metric}",
                {"Train": t_metric, "Val": v_metric},
                epoch,
            )

        for item in self.optimizers_and_schedulers:
            optimizer = item["optimizer"]
            optimizer_name = optimizer.__class__.__name__
            lr_group = {}
            for i, param_group in enumerate(optimizer.param_groups):
                lr_group[f"g{i}"] = param_group["lr"]

            self.tensorboard_writer.add_scalars(
                f"LR/{optimizer_name}",
                lr_group,
                epoch,
            )

        self.tensorboard_writer.flush()

    def _update_schedulers(
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
        if not self.optimizers_and_schedulers:
            return

        for item in self.optimizers_and_schedulers:
            scheduler = item["scheduler"]
            if scheduler is None:
                continue

            if epoch and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if val_loss is not None:
                    scheduler.step(val_loss)
                else:
                    self.logger.warning(
                        "Couldn't update ReduceLROnPlateau scheduler because val_loss is None when epoch update was requested."
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

    def _update_metrics(self, logits: torch.Tensor, targets: torch.Tensor):
        """Updates each evaluation metric"""
        for metric in self.metrics.values():
            metric.update(
                # reshape to [batch_size * channels, time]
                logits.view(logits.shape[0], -1).cpu().detach(),
                targets.view(targets.shape[0], -1).cpu().detach(),
            )

    def _compute_metrics(self) -> dict:
        """
        Computes each evaluation metric.
        Evaluation metrics are updated per batch in both training and validation.

        Returns:
        - Dictionary containing names of metrics defined in the configuration and their corresponding results
        """
        self.logger.info("Computing metrics")
        results = {}
        for m, f in self.metrics.items():
            results[m] = f.compute()
            f.reset()

        return results

    def _compute_weighted_losses(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        weighted_losses = {}

        for name, criterion in self.criterions.items():
            raw_loss = criterion(logits, targets)

            weighted_loss = raw_loss * self.criterions_stats[name]["weight"]
            weighted_losses[name] = weighted_loss

        # sum weighted individual losses to get the total loss for this batch
        total_weighted_loss = sum(weighted_losses.values())

        return total_weighted_loss, weighted_losses

    def _accumulate_and_step(self, loss: torch.Tensor, step: int, total_steps: int):
        # clear gradients
        if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == total_steps:
            for item in self.optimizers_and_schedulers:
                item["optimizer"].zero_grad()

        # perform backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # if reached grad_accum_steps, step schedulers & optimizer and update scaler
        if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == total_steps:
            for item in self.optimizers_and_schedulers:
                optimizer = item["optimizer"]
                if self.clip_norm:
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.model.parameters(), self.clip_norm)

                if self.scaler:
                    self.scaler.step(optimizer)
                else:
                    optimizer.step()

            if self.scaler:
                self.scaler.update()

    def _train_one_epoch(self, data_loader: DataLoader) -> dict:
        """
        Args:
            data_loader: DataLoader for training data.
        Returns:
        - Epoch training metrics. Check compute_metrics() for more details.
        """
        self.model.train()
        running_total_weighted_loss = 0.0
        # initialize running sums for individual weighted losses for reporting
        running_weighted_losses = {name: 0.0 for name in self.criterions.keys()}

        autocast_enabled = self.scaler is not None

        total_steps = len(data_loader)
        pbar = tqdm(data_loader, total=total_steps, desc="Training")

        for step, (_, eid, inputs, targets) in enumerate(pbar):
            eid, inputs, targets = (
                eid.to(self.device),
                inputs.to(self.device),
                targets.to(self.device),
            )
            with autocast(device_type=self.device.type, enabled=autocast_enabled):
                logits = self.model((eid, inputs))

                total_weighted_loss, weighted_losses = self._compute_weighted_losses(
                    logits, targets
                )

            # backward pass
            self._accumulate_and_step(total_weighted_loss, step, total_steps)

            # accumulate loss for reporting
            running_total_weighted_loss += total_weighted_loss.item()
            for name, loss_value in weighted_losses.items():
                running_weighted_losses[name] += loss_value.item()

            # update schedulers for STEP only
            self._update_schedulers(step=True, epoch=False, val_loss=None)

        # normalize accumulated losses by number of batches
        # since each criterion returns avg loss per sample for the batch,
        # summing these up and dividing by num_batches effectively gives
        # the average total weighted loss per sample over the epoch.
        num_batches = len(data_loader)
        avg_total_weighted_loss = running_total_weighted_loss / num_batches
        avg_individual_weighted_losses = {
            name: loss / num_batches for name, loss in running_weighted_losses.items()
        }

        return {
            "Losses": {
                **avg_individual_weighted_losses,
                "Total": avg_total_weighted_loss,
            },
            "Metrics": self._compute_metrics(),
        }

    def train(
        self,
        max_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
    ):
        """
        Train the model over multiple epochs. The last checkpoint is automatically loaded.
        Args:
            max_epochs: maximum number of epochs to train the model for.
            train_loader: training set DataLoader.
            val_loader: validation set DataLoader.
            test_loader: optional test set DataLoader.
        """
        # create logs dir and tensorboard self.loggers for current training run
        self._open_writer()

        # load last checkpoint (or start from scratch if 1)
        start_epoch = self.checkpointer.load_checkpoint(self.device)

        # log training info
        self.logger.info(
            f"Starting training from epoch: {start_epoch} for {max_epochs} max epochs."
        )

        train_metrics = {}
        val_metrics = {}
        epoch = start_epoch
        try:
            for epoch in range(start_epoch, max_epochs):
                self.logger.info(f"Entering Epoch: {epoch}/{max_epochs}")
                train_metrics = self._train_one_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)

                if test_loader and epoch in self.test_epochs:
                    self.test(test_loader)

                val_loss = val_metrics["Losses"]["Total"]
                if epoch >= self.start_scheduling:
                    self._update_schedulers(step=False, epoch=True, val_loss=val_loss)
                else:
                    self.logger.info(
                        f"Current epoch {epoch} < {self.start_scheduling}, skipping scheduling"
                    )

                self._log_tensorboard_metrics(epoch, train_metrics, val_metrics)
                self.checkpointer.save_checkpoint(epoch, val_loss=val_loss)

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted. Saving checkpoint and exiting")
            self.checkpointer.save_checkpoint(epoch)

        finally:
            self.logger.info("Finished training")
            self._close_writer()

    def evaluate(
        self, data_loader: DataLoader, output_dir: Optional[Path] = None
    ) -> dict:
        """
        Args:
            data_loader: validation set DataLoader.
            output_dir: optional directory to save intermediate outputs
        Returns:
        - Evaluation metrics. Check compute_metrics() for more details.
        """
        self.model.eval()
        with torch.no_grad():
            running_total_weighted_loss = 0.0
            running_individual_weighted_losses = {
                name: 0.0 for name in self.criterions.keys()
            }

            pbar = tqdm(data_loader, total=len(data_loader), desc="Evaluation")
            for id, eid, inputs, targets in pbar:
                eid, inputs, targets = (
                    eid.to(self.device),
                    inputs.to(self.device),
                    targets.to(self.device),
                )
                logits = self.model((eid, inputs))

                if output_dir is not None:
                    # ensure logits are on CPU and detached before saving
                    save_audio(
                        logits.cpu().detach(), data_loader.dataset.fs, id, output_dir
                    )

                total_weighted_loss, weighted_losses = self._compute_weighted_losses(
                    logits, targets
                )

                # accumulate for reporting
                running_total_weighted_loss += total_weighted_loss.item()
                for name, loss_value in weighted_losses.items():
                    running_individual_weighted_losses[name] += loss_value.item()

                self._update_metrics(logits, targets)

            # normalize accumulated losses by number of batches
            num_batches = len(data_loader)
            avg_total_weighted_loss = running_total_weighted_loss / num_batches
            avg_individual_weighted_losses = {
                name: loss / num_batches
                for name, loss in running_individual_weighted_losses.items()
            }

            return {
                "Losses": {
                    **avg_individual_weighted_losses,
                    "Total": avg_total_weighted_loss,
                },
                "Metrics": self._compute_metrics(),
            }

    def test(self, test_loader: DataLoader, output_dir: Optional[Path] = None):
        """
        Evaluate the model on the test data
        Args:
            test_loader: test DataLoader.
        """
        self.logger.info("Testing model")
        if output_dir is not None:
            output_dir.mkdir(exist_ok=True)
            self.logger.info(f"Saving output to: {output_dir}")

        metrics = self.evaluate(test_loader, output_dir=output_dir)

        self.logger.info("=== Test losses ===")
        for k, v in metrics["Losses"].items():
            self.logger.info(f"{k}: {v:.4f}")

        self.logger.info("=== Test metrics ===")
        for k, v in metrics["Metrics"].items():
            self.logger.info(f"{k}: {v:.4f}")

    def _open_writer(self):
        self.logger.info("Opening tensorboard writer")
        self.logs_dir = self.model_dir.joinpath(
            "logs", datetime.now().strftime("%d%m%Y-%H%M%S")
        )

        self.logs_dir.mkdir(exist_ok=False)

        self.tensorboard_writer = SummaryWriter(log_dir=self.logs_dir)

    def _close_writer(self):
        """
        Close the Tensorboard writer.
        """
        self.logger.info("Closing tensorboard writer")
        self.tensorboard_writer.close()
