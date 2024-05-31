import torch
from torch import nn
from tqdm import tqdm

import wandb
from efficient_multilingual_continual_pretraining import logger
from efficient_multilingual_continual_pretraining.metrics import MetricCalculator
from efficient_multilingual_continual_pretraining.utils import generate_device


class Trainer:
    def __init__(
        self,
        use_wandb: bool,
        device: torch.device,
        criterion: torch.nn.Module = nn.CrossEntropyLoss(),
        mode: str = "binary",
    ) -> None:

        self.metric_calculator = MetricCalculator(mode)
        self.criterion = criterion
        self.device = device
        self.use_wandb = use_wandb
        self.device = device if device is not None else generate_device()

        self._watcher_command = wandb.log if self.use_wandb else lambda *_: None

        self.optimizer = None
        self.model = None

    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        n_epochs: int = 10,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        verbose: bool = True,
        watch: bool = True,
    ) -> None:

        for epoch in range(1, n_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{n_epochs}.")
            # Train
            epoch_train_state = self._train_single_epoch(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                verbose=verbose,
            )

            current_epoch_scores = {"train": epoch_train_state}

            if scheduler is not None:
                scheduler.step()

            # Validation
            if val_dataloader is not None:
                epoch_val_scores = self.evaluate(
                    model=model,
                    val_dataloader=val_dataloader,
                    verbose=verbose,
                )
                current_epoch_scores["val"] = epoch_val_scores

            current_epoch_scores["train_epoch"] = epoch
            if watch:
                try:
                    self._watcher_command(current_epoch_scores)
                except Exception as e:
                    logger.error(f"Error loading to watcher after train at epoch {epoch}!")
                    raise e

            logger.info(f"Finished epoch {epoch}/{n_epochs}.")

    def _train_single_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        verbose: bool,
    ) -> dict[str, float]:

        logger.info("Started training the model.")
        model.train()
        pbar = tqdm(train_dataloader, leave=False, desc="Training model") if verbose else train_dataloader
        self.metric_calculator.reset()
        train_loss = 0.0

        for items_batch, targets_batch in pbar:
            items_batch = {key: tensor.to(self.device) for key, tensor in items_batch.items()}
            targets_batch = targets_batch.to(self.device)

            optimizer.zero_grad()
            predicted_logits = model(items_batch, cast_to_probabilities=False)
            loss = self.criterion(predicted_logits, targets_batch)

            # We use running loss and scores to avoid double-running through the train dataset. This does not
            #   provide objective scores, but is good enough compared to time-effectiveness.
            train_loss += loss.item() * len(targets_batch)
            self.metric_calculator.update(predicted_logits, targets_batch)

            loss.backward()
            optimizer.step()

        result = {
            "train_loss": train_loss / len(train_dataloader.dataset),
            **self.metric_calculator.calculate_metrics(),
        }

        logger.info("Finished training the model.")
        logger.debug(f"Model train scores: {result}")

        return result

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> dict[str, float]:

        logger.info("Started validating the model.")
        model.eval()
        pbar = tqdm(val_dataloader, leave=False, desc="Validating model") if verbose else val_dataloader
        self.metric_calculator.reset()
        val_loss = 0.0

        for items_batch, targets_batch in pbar:
            items_batch = {key: tensor.to(self.device) for key, tensor in items_batch.items()}
            targets_batch = targets_batch.to(self.device)

            predicted_logits = model(items_batch, cast_to_probabilities=False)
            loss = self.criterion(predicted_logits, targets_batch)

            val_loss += loss.item() * len(targets_batch)
            self.metric_calculator.update(predicted_logits, targets_batch)

        result = {
            "val_loss": val_loss / len(val_dataloader.dataset),
            **self.metric_calculator.calculate_metrics(),
        }

        logger.info("Finished validating the model.")
        logger.debug(f"Model val scores: {result}")
        return result
