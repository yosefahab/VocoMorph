from pathlib import Path
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from src.trainer.logger import TensorBoardLogger
from src.utils.logger import get_logger

from .checkpointer import Checkpointer
from .factory import get_criterions, get_metrics, get_optimizer


class ModelTrainer:
    """
    A class to handle the training, evaluation, and logging for an MLX model.
    """

    def __init__(self, config: dict, model_dir: Path, model: nn.Module):
        assert model_dir.exists(), f"Model directory {model_dir} does not exist."

        self.logger = get_logger(self.__class__.__name__)

        self.model_dir = model_dir
        self.config = config
        self.model = model

        self.clip_norm = self.config["trainer"].get("clip_norm")

        # Assumes get_criterions returns instantiated loss objects, not functions.
        self.criterions = get_criterions(self.config["criterions"])
        self.criterions_stats: Dict[str, Dict[str, float]] = {
            c["name"]: {"weight": c.get("weight", 1.0)}
            for c in self.config["criterions"]
        }

        # Instantiate MLX optimizer once.
        self.optimizer = get_optimizer(self.config["optimizers"])

        self.test_epochs = self.config["trainer"].get("test_epochs", [])
        self.metrics = get_metrics(self.config["metrics"])

        self.checkpoints_dir = self.model_dir.joinpath("checkpoints")
        self.checkpoints_dir.mkdir(exist_ok=True)

        # The step count will be stored and loaded with checkpoints.
        self.global_step = 0

        self.checkpointer = Checkpointer(
            self.checkpoints_dir, self.config["checkpointer"], self.model
        )
        self._get_model_summary()

    def _get_model_summary(self):
        self.logger.info("Generating model summary")
        with open(self.model_dir.joinpath("model_summary.txt"), "w") as f:
            f.write(self.model.__str__())

    def _compute_weighted_losses(
        self, logits: mx.array, targets: mx.array
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Computes the total weighted loss and a dictionary of individual losses.
        """
        weighted_losses: Dict[str, mx.array] = {}
        for name, criterion in self.criterions.items():
            # This is the corrected line where the __call__ method of the
            # instantiated loss object is invoked.
            raw_loss: mx.array = criterion(logits, targets)
            weighted_loss = raw_loss * self.criterions_stats[name]["weight"]
            weighted_losses[name] = weighted_loss

        total_weighted_loss = mx.sum(mx.stack(list(weighted_losses.values())))
        return total_weighted_loss, weighted_losses

    def run_epoch(self, data_stream) -> dict:
        """
        Runs a single training epoch.
        """

        # The loss function now returns both the total loss and individual weighted losses.
        def loss_fn(model, effect_id, inputs, targets):
            logits = model(effect_id, inputs)
            total_weighted_loss, weighted_losses = self._compute_weighted_losses(
                logits, targets
            )
            return total_weighted_loss, (logits, weighted_losses)

        # Use value_and_grad to get both the loss and the gradients.
        grad_fn = nn.value_and_grad(self.model, loss_fn)

        running_total_weighted_loss = 0.0
        running_weighted_losses = dict.fromkeys(self.criterions, 0.0)
        pbar = tqdm(data_stream, desc="Training")

        for batch in pbar:
            effect_id = mx.array(batch["effect_id"])
            inputs = mx.array(batch["inputs"])
            targets = mx.array(batch["targets"])

            # The grad_fn now returns loss, grads, and the extra tuple from loss_fn.
            (loss, (_, weighted_losses)), grads = grad_fn(
                self.model.state, effect_id, inputs, targets
            )

            # Update the optimizer with the model's parameters and the computed gradients.
            self.optimizer.update(self.model.parameters(), grads)
            mx.eval(self.model.state, self.optimizer.state)

            self.global_step += 1
            running_total_weighted_loss += loss.item()

            for name, loss_value in weighted_losses.items():
                running_weighted_losses[name] += loss_value.item()

            pbar.set_description(
                f"Training Loss: {running_total_weighted_loss / (pbar.n + 1):.4f}"
            )

        num_batches = len(data_stream)
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

    def train(self, max_epochs: int, train_stream, val_stream, test_stream):
        experiment_logger = TensorBoardLogger(self.model_dir.joinpath("logs"))
        start_epoch = self.checkpointer.load_checkpoint()
        self.logger.info(
            f"Starting training from epoch: {start_epoch} for {max_epochs} max epochs."
        )
        val_loss = 0
        epoch = start_epoch
        try:
            for epoch in range(start_epoch, max_epochs):
                self.logger.info(f"Entering Epoch: {epoch}/{max_epochs}")
                train_metrics = self.run_epoch(train_stream)
                val_metrics = self.evaluate(val_stream)

                if epoch in self.test_epochs and test_stream:
                    self.test(test_stream)

                val_loss = val_metrics["Losses"]["Total"]

                experiment_logger.log_metrics(epoch, train_metrics, val_metrics)
                self.checkpointer.save_checkpoint(epoch, val_loss=val_loss)
        except KeyboardInterrupt:
            self.logger.warning("Trainign interrupted. Saving checkpoint and exitting.")
            self.checkpointer.save_checkpoint(epoch, val_loss=val_loss)

        finally:
            experiment_logger.close_writer()
            self.logger.info("Finished training")

    def evaluate(self, data_stream, output_dir: Optional[Path] = None) -> dict:
        self.model.eval()

        running_total_weighted_loss = 0.0
        running_individual_weighted_losses = {
            name: 0.0 for name in self.criterions.keys()
        }
        pbar = tqdm(data_stream, desc="Evaluation")

        for batch in pbar:
            effect_id = batch["effect_id"]
            inputs = batch["inputs"]
            targets = batch["targets"]

            # Corrected: Pass effect_id and inputs as separate arguments.
            logits = self.model(effect_id, inputs)

            # if output_dir is not None:
            #     save_audio(logits, 16_000, item_ids, output_dir)

            total_weighted_loss, weighted_losses = self._compute_weighted_losses(
                logits, targets
            )
            running_total_weighted_loss += total_weighted_loss.item()
            for name, loss_value in weighted_losses.items():
                running_individual_weighted_losses[name] += loss_value.item()

            self._update_metrics(logits, targets)

        num_batches = len(data_stream)
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

    def _update_metrics(self, logits: mx.array, targets: mx.array):
        """Updates each evaluation metric"""
        # The .asnumpy() call is fine here for compatibility with other libraries.
        for metric in self.metrics.values():
            metric.update(
                mx.reshape(logits, (logits.shape[0], -1)).asnumpy(),
                mx.reshape(targets, (targets.shape[0], -1)).asnumpy(),
            )

    def _compute_metrics(self) -> dict:
        """Computes each evaluation metric."""
        self.logger.info("Computing metrics")
        results = {}
        for m, f in self.metrics.items():
            results[m] = f.compute()
            f.reset()
        return results

    def test(self, test_stream, output_dir: Optional[Path] = None):
        self.logger.info("Testing model")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Saving output to: {output_dir}")

        metrics = self.evaluate(test_stream, output_dir=output_dir)

        self.logger.info("=== Test losses ===")
        for k, v in metrics["Losses"].items():
            self.logger.info(f"{k}: {v:.4f}")

        self.logger.info("=== Test metrics ===")
        for k, v in metrics["Metrics"].items():
            self.logger.info(f"{k}: {v:.4f}")
