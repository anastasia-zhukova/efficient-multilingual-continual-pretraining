from typing import Literal

import torch

from efficient_multilingual_continual_pretraining import logger


class MetricCalculator:
    def __init__(
        self,
        device: torch.device,
        mode: Literal["binary", "multi-class", "multi-label"] = "binary",
        n_classes: int = 2,
    ):
        self.device = device
        self.mode = mode
        if self.mode == "binary":
            self.n_classes = 1
        else:
            self.n_classes = n_classes

        self.true_positives = torch.zeros(self.n_classes, device=self.device)
        self.false_positives = torch.zeros(self.n_classes, device=self.device)
        self.false_negatives = torch.zeros(self.n_classes, device=self.device)
        self.true_negatives = torch.zeros(self.n_classes, device=self.device)

    def calculate_metrics(self) -> dict:
        return {
            "accuracy": self._calculate_accuracy(),
            "precision": self._calculate_precision(),
            "recall": self._calculate_recall(),
            "f1": self._calculate_f1(),
        }

    def reset(self):
        self.true_positives = torch.zeros(self.n_classes, device=self.device)
        self.false_positives = torch.zeros(self.n_classes, device=self.device)
        self.false_negatives = torch.zeros(self.n_classes, device=self.device)
        self.true_negatives = torch.zeros(self.n_classes, device=self.device)

    def update(
        self,
        predicted_logits: torch.Tensor,
        actual_classes: torch.Tensor,
    ) -> None:
        if self.mode == "multi-label":
            predicted_probabilities = torch.sigmoid(predicted_logits)
            predicted_classes = predicted_probabilities > 0.5
            actual_positive = actual_classes == 1

            self.true_positives += ((predicted_classes == actual_classes) & actual_positive).sum(dim=0)
            self.true_negatives += ((predicted_classes == actual_classes) & ~actual_positive).sum(dim=0)
            self.false_negatives += ((predicted_classes != actual_classes) & actual_positive).sum(dim=0)
            self.false_positives += ((predicted_classes != actual_classes) & ~actual_positive).sum(dim=0)

        elif self.mode == "multi-class":
            # TODO: review this, it appears to be non-effective.
            max_logit_values, _ = torch.max(predicted_logits, dim=1, keepdim=True)
            predicted_classes = predicted_logits == max_logit_values
            actual_classes = actual_classes.int()
            sparse_classes = torch.zeros(len(actual_classes), self.n_classes)
            sparse_classes[torch.arange(len(sparse_classes)), actual_classes] = 1
            actual_positive = sparse_classes == 1

            self.true_positives += ((predicted_classes == actual_classes) & actual_positive).sum(dim=0)
            self.true_negatives += ((predicted_classes == actual_classes) & ~actual_positive).sum(dim=0)
            self.false_negatives += ((predicted_classes != actual_classes) & actual_positive).sum(dim=0)
            self.false_positives += ((predicted_classes != actual_classes) & ~actual_positive).sum(dim=0)

        elif self.mode == "binary":
            predicted_classes = torch.argmax(predicted_logits, dim=1)
            actual_positive = actual_classes == 1

            self.true_positives += ((predicted_classes == actual_classes) & actual_positive).sum(dim=0)
            self.true_negatives += ((predicted_classes == actual_classes) & ~actual_positive).sum(dim=0)
            self.false_negatives += ((predicted_classes != actual_classes) & actual_positive).sum(dim=0)
            self.false_positives += ((predicted_classes != actual_classes) & ~actual_positive).sum(dim=0)

    # We adopt macroaveraging since it is said it better deals with class imbalance.
    #   https://education.yandex.ru/handbook/ml/article/metriki-klassifikacii-i-regressii
    def _calculate_accuracy(self):
        denominator = self.true_positives + self.true_negatives + self.false_negatives + self.false_positives
        if (denominator == 0).any():
            logger.warning("Detected division by zero when calculating accuracy score!")
            return 0
        numerator = self.true_positives + self.true_negatives
        scores = numerator / denominator
        return float(scores.mean())

    def _calculate_precision(self) -> float:
        denominator = self.true_positives + self.false_positives

        zero_denominator_mask = denominator == 0
        if zero_denominator_mask.any():
            logger.warning("Detected division by zero when calculating precision!")

        precision_per_label = torch.where(
            zero_denominator_mask,
            torch.zeros_like(denominator),
            self.true_positives / denominator,
        )

        return float(precision_per_label.mean())

    def _calculate_recall(self) -> float:
        denominator = self.true_positives + self.false_negatives

        zero_denominator_mask = denominator == 0
        if zero_denominator_mask.any():
            logger.warning("Detected division by zero when calculating precision!")

        recall_per_label = torch.where(
            zero_denominator_mask,
            torch.zeros_like(denominator),
            self.true_positives / denominator,
        )
        return float(recall_per_label.mean())

    def _calculate_f1(self) -> float:
        denominator = 2 * self.true_positives + self.false_negatives + self.false_positives

        zero_denominator_mask = denominator == 0
        if zero_denominator_mask.any():
            logger.warning("Detected division by zero when calculating precision!")

        f1_per_label = torch.where(
            zero_denominator_mask,
            torch.zeros_like(denominator),
            2 * self.true_positives / denominator,
        )
        return float(f1_per_label.mean())
