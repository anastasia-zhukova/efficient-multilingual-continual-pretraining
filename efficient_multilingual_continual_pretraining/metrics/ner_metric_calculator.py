import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

from efficient_multilingual_continual_pretraining import logger


class NERMetricCalculator:
    def __init__(self, id_to_token_mapping: dict):
        self.id_to_token_mapping = id_to_token_mapping
        self.model_predictions = []
        self.true_labels = []

    def calculate_metrics(self) -> dict:
        unique = set(item for sublist in self.model_predictions for item in sublist)
        logger.debug(f"Found {len(unique)} unique entities in predictions: {unique}.")
        precision = precision_score(
            self.true_labels,
            self.model_predictions,
            mode="strict",
            scheme=IOB2,
            average="micro",
        )
        recall = recall_score(
            self.true_labels,
            self.model_predictions,
            mode="strict",
            scheme=IOB2,
            average="micro",
        )
        f1 = f1_score(
            self.true_labels,
            self.model_predictions,
            mode="strict",
            scheme=IOB2,
            average="micro",
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def reset(self) -> None:
        self.model_predictions = []
        self.true_labels = []

    def update(
        self,
        predicted_logits: torch.Tensor,
        actual_classes: torch.Tensor,
    ) -> None:
        predicted_logits = predicted_logits.detach().cpu()
        actual_classes = actual_classes.detach().cpu()
        predicted_classes = torch.argmax(predicted_logits, dim=2)

        predictions, labels = [], []
        for prediction, label in zip(predicted_classes, actual_classes, strict=True):
            filtered_predictions = []
            filtered_labels = []
            for token_predictions, true_token_label in zip(prediction, label, strict=True):
                if true_token_label != -100:
                    filtered_predictions.append(self.id_to_token_mapping[token_predictions.item()])
                    filtered_labels.append(self.id_to_token_mapping[true_token_label.item()])
            predictions.append(filtered_predictions)
            labels.append(filtered_labels)

        self.model_predictions.extend(predictions)
        self.true_labels.extend(labels)
