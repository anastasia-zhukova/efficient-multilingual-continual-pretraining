from typing import Literal

import torch
from torch import nn
from tqdm import tqdm

import wandb
from efficient_multilingual_continual_pretraining import logger
from efficient_multilingual_continual_pretraining.metrics import MetricCalculator, NERMetricCalculator
from efficient_multilingual_continual_pretraining.utils import generate_device, verbose_iterator, log_with_message


class BaseTrainer:
    def __init__(
        self,
        use_wandb: bool,
        device: torch.device,
        criterion: torch.nn.Module = nn.CrossEntropyLoss(),
        mode: Literal["binary", "multi-class", "multi-label", "NER"] = "binary",
        n_classes: int | None = None,
        id_to_token_mapping: dict | None = None,
    ) -> None:

        self.mode = mode
        if mode == "NER":
            if id_to_token_mapping is None:
                raise ValueError("No id_to_token_mapping provided for NER task!")
            self.metric_calculator = NERMetricCalculator(id_to_token_mapping)
        else:
            self.metric_calculator = MetricCalculator(device, mode, n_classes)

        self.criterion = criterion
        self.device = device
        self.use_wandb = use_wandb
        self.device = device if device is not None else generate_device()
        self.id_to_token_mapping = id_to_token_mapping

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

            logger.debug(f"Current scores: {current_epoch_scores}")
            current_epoch_scores["train_epoch"] = epoch
            if watch:
                try:
                    self._watcher_command(current_epoch_scores)
                except Exception as e:
                    logger.error(f"Error loading to watcher after train at epoch {epoch}!")
                    raise e

            logger.info(f"Finished epoch {epoch}/{n_epochs}.")

    @log_with_message("pretraining the model")
    def pretrain(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        test_dataloader: torch.utils.data.DataLoader | None = None,
        n_steps: int = 10_000,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        verbose: bool = True,
        watch: bool = True,
        steps_to_log: int = 500,
    ) -> None:
        model.train()
        current_step = 0
        train_loss = 0.0
        total_loss_items = 0

        while current_step < n_steps:
            for batch in train_dataloader:
                if current_step >= n_steps:
                    break
                current_step += 1

                # Train steps
                items_batch = {key: tensor.to(self.device) for key, tensor in batch.items()}
                optimizer.zero_grad()
                outputs = model(**items_batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # Loss logging handling
                n_loss_objects = len(items_batch["labels"])
                train_loss += loss.item() * n_loss_objects
                total_loss_items += n_loss_objects

                # In case it is time to log
                if current_step % steps_to_log == 0:
                    log_dict = {
                        "train": {"loss": train_loss / total_loss_items},
                        "step": current_step,
                    }
                    if val_dataloader is not None:
                        log_dict["val"] = {"loss": self.evaluate_pretrain(model, val_dataloader, verbose)}
                    logger.debug(f"Reached logging checkpoint at step {current_step}, current scores: {log_dict}")

                    self._safe_watcher_log(
                        watch,
                        log_dict,
                        f"Error loading to watcher after pretrain at step {current_step}!",
                    )

                    # Reset running values
                    train_loss = 0
                    total_loss_items = 0
                    model.train()

            logger.info(f"Exhausted train dataset, current step {current_step}.")
            if scheduler is not None:
                scheduler.step()

        if val_dataloader is not None or test_dataloader is not None:
            log_dict = {
                "step": current_step,
            }
            if val_dataloader is not None:
                log_dict["val"] = {"loss": self.evaluate_pretrain(model, val_dataloader, verbose)}
            if test_dataloader is not None:
                log_dict["test"] = {"loss": self.evaluate_pretrain(model, test_dataloader, verbose)}
            logger.debug(f"Reached the final checkpoint at step {current_step}, final scores: {log_dict}")

            self._safe_watcher_log(watch, log_dict, f"Error loading to watcher after pretrain at step {current_step}!")

    # def _pretrain_single_epoch(
    #     self,
    #     model: torch.nn.Module,
    #     optimizer: torch.optim.Optimizer,
    #     train_dataloader: torch.utils.data.DataLoader,
    #     verbose: bool,
    # ) -> dict:
    #     logger.info("Started pretraining the model.")
    #     model.train()
    #     pbar = verbose_iterator(, verbose, leave=False, desc="Training model")
    #     self.metric_calculator.reset()
    #     train_loss = 0.0
    #     total_loss_items = 0
    #     for batch in pbar:
    #         items_batch = {key: tensor.to(self.device) for key, tensor in batch.items()}
    #         optimizer.zero_grad()
    #         outputs = model(**items_batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #
    #         n_loss_objects = len(items_batch["labels"])
    #         train_loss += loss.item() * n_loss_objects
    #         total_loss_items += n_loss_objects
    #
    #     result = {"train_loss": train_loss / total_loss_items}
    #
    #     logger.info("Finished training the model.")
    #     logger.info(f"Model train scores: {result}")
    #     return result

    def _train_single_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        verbose: bool,
    ) -> dict[str, float]:

        logger.info("Started training the model.")
        model.train()
        pbar = verbose_iterator(train_dataloader, verbose, leave=False, desc="Training model")
        self.metric_calculator.reset()
        train_loss = 0.0
        total_loss_items = 0

        for batch in pbar:
            items_batch = {
                key: tensor.to(self.device)
                for key, tensor in batch.items()
                if key not in ["targets", "cast_to_probabilities", "paragraph_tokens"]
            }
            if "paragraph_tokens" in batch:
                items_batch["paragraph_tokens"] = batch["paragraph_tokens"]

            targets_batch = batch["targets"].to(self.device)

            optimizer.zero_grad()
            predicted_logits = model(**items_batch, cast_to_probabilities=batch.get("cast_to_probabilities", False))

            if self.mode == "NER":
                viewed_logits = predicted_logits.view(-1, len(self.id_to_token_mapping))
                n_loss_objects = len(viewed_logits)
                loss = self.criterion(viewed_logits, targets_batch.view(-1))
            else:
                n_loss_objects = len(targets_batch)
                loss = self.criterion(predicted_logits, targets_batch)

            # We use running loss and scores to avoid double-running through the train dataset. This does not
            #   provide objective scores, but is good enough compared to time-effectiveness.
            train_loss += loss.item() * n_loss_objects
            self.metric_calculator.update(predicted_logits, targets_batch)

            loss.backward()
            optimizer.step()
            total_loss_items += n_loss_objects

        result = {
            "train_loss": train_loss / total_loss_items,
            **self.metric_calculator.calculate_metrics(),
        }

        logger.info("Finished training the model.")
        logger.info(f"Model train scores: {result}")

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
        total_loss_items = 0

        for batch in pbar:
            items_batch = {
                key: tensor.to(self.device)
                for key, tensor in batch.items()
                if key not in ["targets", "cast_to_probabilities", "paragraph_tokens"]
            }
            if "paragraph_tokens" in batch:
                items_batch["paragraph_tokens"] = batch["paragraph_tokens"]

            targets_batch = batch["targets"].to(self.device)

            predicted_logits = model(**items_batch, cast_to_probabilities=batch.get("cast_to_probabilities", False))

            if self.mode == "NER":
                viewed_logits = predicted_logits.view(-1, len(self.id_to_token_mapping))
                n_loss_objects = len(viewed_logits)
                loss = self.criterion(viewed_logits, targets_batch.view(-1))
            else:
                n_loss_objects = len(targets_batch)
                loss = self.criterion(predicted_logits, targets_batch)

            val_loss += loss.item() * n_loss_objects
            self.metric_calculator.update(predicted_logits, targets_batch)

            total_loss_items += n_loss_objects

        result = {
            "val_loss": val_loss / len(val_dataloader.dataset),
            **self.metric_calculator.calculate_metrics(),
        }

        logger.info("Finished validating the model.")
        logger.info(f"Model val scores: {result}")
        return result

    @log_with_message(
        "evaluating the pretraining model",
    )
    @torch.no_grad()
    def evaluate_pretrain(
        self,
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> float:
        model.eval()
        pbar = tqdm(val_dataloader, leave=False, desc="Validating model") if verbose else val_dataloader
        val_loss = 0.0
        total_loss_items = 0

        for batch in pbar:
            items_batch = {key: tensor.to(self.device) for key, tensor in batch.items()}
            outputs = model(**items_batch)
            loss = outputs.loss

            n_loss_objects = len(items_batch["labels"])
            val_loss += loss.item() * n_loss_objects
            total_loss_items += n_loss_objects

        return val_loss / total_loss_items

    def _safe_watcher_log(
        self,
        watch: bool,
        log_dict: dict,
        logger_message: str,
    ) -> None:
        if watch:
            try:
                self._watcher_command(log_dict)
            except Exception as e:
                logger.error(logger_message)
                raise e
