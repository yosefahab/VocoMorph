from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.types import Tensor
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from src.utils.audio import save_audio
from src.utils.logger import get_logger

from .checkpointer import Checkpointer
from .factory import get_criterions, get_metrics, get_optimizers_and_schedulers


class DistributedModelTrainer:
    def __init__(
        self,
        config: dict,
        model_dir: Path,
        model: torch.nn.Module,
        device: torch.device,
        rank: int = -1,
        world_size: int = -1,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        assert model_dir.exists(), f"Model directory {model_dir} does not exist."
        self.model_dir = model_dir
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = self.world_size > 1

        # Initialize DDP for distributed training
        if self.is_distributed:
            self.logger.info(
                f"Initializing DDP for rank {self.rank} on device {self.device}"
            )
            # The model is wrapped in DDP after moving to device
            self.model = model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[self.device] if self.device.type == "cuda" else None,
                find_unused_parameters=True,
            )  # find_unused_parameters can be True if some module outputs are not used in loss calculation, otherwise False
        else:
            self.model = model.to(self.device)

        self.clip_norm = self.config["trainer"]["clip_norm"]

        self.criterions = get_criterions(self.config["criterions"])
        self.criterions_stats: Dict[str, Dict[str, float]] = {
            c["name"]: {"weight": c.get("weight", 1.0)}
            for c in self.config["criterions"]
        }

        # Pass self.model.module.parameters() if using DDP, otherwise self.model.parameters()
        assert isinstance(self.model.module, torch.nn.Module)
        optimizer_params = (
            self.model.module.parameters()
            if self.is_distributed
            else self.model.parameters()
        )
        self.optimizers_and_schedulers = get_optimizers_and_schedulers(
            self.config["optimizers"], optimizer_params
        )

        self.optimizers: List[torch.optim.Optimizer] = []
        self.schedulers: List[torch.optim.lr_scheduler._LRScheduler] = []
        for optimizer, scheduler in self.optimizers_and_schedulers:
            self.optimizers.append(optimizer)
            if scheduler is not None:
                self.schedulers.append(scheduler)

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
            self.model,  # Pass the DDP wrapped model
            self.optimizers,
            self.schedulers,
            self.scaler,
            self.checkpoints_dir,
            is_distributed=self.is_distributed,  # Pass is_distributed to Checkpointer
        )
        self._get_model_summary()

    def _get_model_summary(self):
        # Only log model summary on rank 0
        if self.rank <= 0:
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

            # For model summary, use self.model.module if DDP is used
            model_to_summarize = (
                self.model.module if self.is_distributed else self.model
            )
            assert isinstance(model_to_summarize, torch.nn.Module)
            model_summary = summary(
                model_to_summarize,
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
            assert len(tensors) > 0
            return tensors if len(tensors) > 1 else tensors[0]
        elif isinstance(dummy_cfg, dict):
            return {k: make_tensor(v) for k, v in dummy_cfg.items()}
        else:
            return make_tensor(dummy_cfg)

    def _log_tensorboard_metrics(
        self, epoch: int, train_metrics: dict, val_metrics: dict
    ):
        # Only log on rank 0
        if self.rank <= 0:
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

                assert self.tensorboard_writer is not None
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

            for optimizer, _ in self.optimizers_and_schedulers:
                lr_group = {}
                for i, param_group in enumerate(optimizer.param_groups):
                    lr_group[f"g{i}"] = param_group["lr"]

                self.tensorboard_writer.add_scalars(
                    f"LR/{optimizer.__class__.__name__}",
                    lr_group,
                    epoch,
                )

            self.tensorboard_writer.flush()

    def _update_schedulers(
        self, step: bool = False, epoch: bool = False, val_loss: Optional[float] = None
    ):
        """
        Update learning rate schedulers.
        Args:
        - step: If True, update step-based schedulers (called per batch).
        - epoch: If True, update epoch-based schedulers (called per epoch).
        - val_loss: (Optional) Validation loss for ReduceLROnPlateau.
        """
        for _, scheduler in self.optimizers_and_schedulers:
            if scheduler is None:
                continue

            if epoch and isinstance(
                scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                if val_loss is not None:
                    scheduler.step(val_loss)
                else:
                    self.logger.warning(
                        "Couldn't update ReduceLROnPlateau scheduler because val_loss is None."
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

    def _update_metrics(self, logits: Tensor, targets: Tensor):
        """Updates each evaluation metric"""
        # Metrics are updated locally, then gathered for aggregation
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
            metric_val = f.compute()
            # If distributed, gather metrics from all processes
            if self.is_distributed:
                # Assuming metric_val is a single tensor or can be converted to one
                gathered_metrics = [
                    torch.zeros_like(metric_val).to(self.device)
                    for _ in range(self.world_size)
                ]
                dist.all_gather(gathered_metrics, metric_val.to(self.device))
                # Average or sum the gathered metrics based on how the metric should be combined
                results[m] = (
                    torch.stack(gathered_metrics).mean().item()
                )  # Example: taking mean
            else:
                results[m] = metric_val.item()
            f.reset()

        return results

    def _compute_weighted_losses(
        self, logits: Tensor, targets: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        weighted_losses: dict[str, Tensor] = {}

        for name, criterion in self.criterions.items():
            raw_loss: Tensor = criterion(logits, targets)
            weighted_loss = raw_loss * self.criterions_stats[name]["weight"]
            weighted_losses[name] = weighted_loss

        # sum weighted individual losses to get the total loss for this batch
        total_weighted_loss = sum(weighted_losses.values())

        assert isinstance(total_weighted_loss, Tensor)
        return total_weighted_loss, weighted_losses

    def _accumulate_and_step(self, loss: Tensor, step: int, total_steps: int):
        # Scale loss by grad_accum_steps and world_size for distributed training
        # This ensures that the effective batch size is what you expect
        loss = loss / self.grad_accum_steps
        if self.is_distributed:
            loss = (
                loss / self.world_size
            )  # Divide by world_size to average gradients across processes

        # clear gradients
        if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == total_steps:
            for optimizer, _ in self.optimizers_and_schedulers:
                optimizer.zero_grad()

        # perform backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # if reached grad_accum_steps, step schedulers & optimizer and update scaler
        if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == total_steps:
            for optimizer, _ in self.optimizers_and_schedulers:
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

    def _train_one_epoch(self, data_loader: DataLoader, epoch: int) -> dict:
        """
        Args:
        - data_loader: DataLoader for training data.
        Returns:
            Epoch training metrics. Check compute_metrics() for more details.
        """
        self.model.train()
        running_total_weighted_loss = 0.0
        # initialize running sums for individual weighted losses for reporting
        running_weighted_losses = {name: 0.0 for name in self.criterions.keys()}

        autocast_enabled = self.scaler is not None

        # Set epoch for DistributedSampler to ensure proper shuffling
        if self.is_distributed and isinstance(data_loader.sampler, DistributedSampler):
            data_loader.sampler.set_epoch(epoch)

        total_steps = len(data_loader)
        pbar = tqdm(
            data_loader, total=total_steps, desc="Training", disable=(self.rank != 0)
        )  # Only show progress bar on rank 0

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
            # It's better to reduce losses across all processes before accumulating for reporting
            if self.is_distributed:
                dist.all_reduce(total_weighted_loss, op=dist.ReduceOp.SUM)
                for name in weighted_losses:
                    dist.all_reduce(weighted_losses[name], op=dist.ReduceOp.SUM)

            running_total_weighted_loss += total_weighted_loss.item()
            for name, loss_value in weighted_losses.items():
                running_weighted_losses[name] += loss_value.item()

            # update schedulers for STEP only
            self._update_schedulers(step=True, epoch=False, val_loss=None)

        # normalize accumulated losses by number of batches
        # Since losses were summed across processes, divide by world_size as well
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
        - max_epochs: maximum number of epochs to train the model for.
        - train_loader: training set DataLoader.
        - val_loader: validation set DataLoader.
        - test_loader: optional test set DataLoader.
        """
        # create logs dir and tensorboard self.loggers for current training run
        self._open_writer()

        # load last checkpoint (or start from scratch if 1)
        start_epoch = self.checkpointer.load_checkpoint(self.device)

        # log training info on rank 0
        if self.rank <= 0:
            self.logger.info(
                f"Starting training from epoch: {start_epoch} for {max_epochs} max epochs."
            )

        train_metrics = {}
        val_metrics = {}
        epoch = start_epoch
        try:
            for epoch in range(start_epoch, max_epochs):
                if self.rank <= 0:
                    self.logger.info(f"Entering Epoch: {epoch}/{max_epochs}")

                # If using DistributedSampler, set the epoch for shuffling
                if self.is_distributed and isinstance(
                    train_loader.sampler, DistributedSampler
                ):
                    train_loader.sampler.set_epoch(epoch)
                if self.is_distributed and isinstance(
                    val_loader.sampler, DistributedSampler
                ):
                    val_loader.sampler.set_epoch(epoch)
                if (
                    test_loader
                    and self.is_distributed
                    and isinstance(test_loader.sampler, DistributedSampler)
                ):
                    test_loader.sampler.set_epoch(epoch)

                train_metrics = self._train_one_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)

                if test_loader and epoch in self.test_epochs:
                    self.test(test_loader)

                val_loss = val_metrics["Losses"]["Total"]
                if epoch >= self.start_scheduling:
                    self._update_schedulers(step=False, epoch=True, val_loss=val_loss)
                else:
                    if self.rank <= 0:
                        self.logger.info(
                            f"Current epoch {epoch} < {self.start_scheduling}, skipping scheduling"
                        )

                self._log_tensorboard_metrics(epoch, train_metrics, val_metrics)
                self.checkpointer.save_checkpoint(epoch, val_loss=val_loss)

        except KeyboardInterrupt:
            if self.rank <= 0:
                self.logger.warning(
                    "Training interrupted. Saving checkpoint and exiting"
                )
            self.checkpointer.save_checkpoint(epoch)

        finally:
            if self.rank <= 0:
                self.logger.info("Finished training")
            self._close_writer()

    def evaluate(
        self, data_loader: DataLoader, output_dir: Optional[Path] = None
    ) -> dict:
        """
        Args:
        - data_loader: validation set DataLoader.
        - output_dir: optional directory to save intermediate outputs
        Returns:
            Evaluation metrics. Check compute_metrics() for more details.
        """
        self.model.eval()
        with torch.no_grad():
            running_total_weighted_loss = 0.0
            running_individual_weighted_losses = {
                name: 0.0 for name in self.criterions.keys()
            }

            pbar = tqdm(
                data_loader,
                total=len(data_loader),
                desc="Evaluation",
                disable=(self.rank != 0),
            )
            for id, eid, inputs, targets in pbar:
                eid, inputs, targets = (
                    eid.to(self.device),
                    inputs.to(self.device),
                    targets.to(self.device),
                )
                logits = self.model((eid, inputs))

                if (
                    output_dir is not None and self.rank <= 0
                ):  # Only save outputs on rank 0
                    # ensure logits are on CPU and detached before saving
                    save_audio(
                        logits.cpu().detach(),
                        data_loader.dataset.fs,  # pyright: ignore[reportAttributeAccessIssue]
                        id,
                        output_dir,
                    )

                total_weighted_loss, weighted_losses = self._compute_weighted_losses(
                    logits, targets
                )

                # accumulate for reporting
                # If distributed, reduce losses across processes before accumulating
                if self.is_distributed:
                    dist.all_reduce(total_weighted_loss, op=dist.ReduceOp.SUM)
                    for name in weighted_losses:
                        dist.all_reduce(weighted_losses[name], op=dist.ReduceOp.SUM)

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
        - test_loader: test DataLoader.
        """
        if self.rank <= 0:
            self.logger.info("Testing model")
            if output_dir is not None:
                output_dir.mkdir(exist_ok=True)
                self.logger.info(f"Saving output to: {output_dir}")

        metrics = self.evaluate(test_loader, output_dir=output_dir)

        # Only log test results on rank 0
        if self.rank <= 0:
            self.logger.info("=== Test losses ===")
            for k, v in metrics["Losses"].items():
                self.logger.info(f"{k}: {v:.4f}")

            self.logger.info("=== Test metrics ===")
            for k, v in metrics["Metrics"].items():
                self.logger.info(f"{k}: {v:.4f}")

    def _open_writer(self):
        # Only open writer on rank 0
        if self.rank <= 0:
            self.logger.info("Opening tensorboard writer")
            self.logs_dir = self.model_dir.joinpath(
                "logs", datetime.now().strftime("%d%m%Y-%H%M%S")
            )

            self.logs_dir.mkdir(parents=True, exist_ok=False)

            self.tensorboard_writer = SummaryWriter(log_dir=self.logs_dir)
        else:
            self.tensorboard_writer = None  # Set to None for other ranks

    def _close_writer(self):
        """
        Close the Tensorboard writer.
        """
        # Only close writer on rank 0
        if self.rank <= 0 and self.tensorboard_writer is not None:
            self.logger.info("Closing tensorboard writer")
            self.tensorboard_writer.close()
