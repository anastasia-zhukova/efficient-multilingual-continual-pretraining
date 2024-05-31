import numpy as np
import torch


class MetricCalculator:
    def __init__(self, mode="binary"):
        self.mode = mode
        self.true_positives = []
        self.false_positives = []
        self.false_negatives = []
        self.true_negatives = []
        self.total_labels = 0
        self.total_samples = 0
        self.correct_predictions = 0

    def update(self, predicted_logits: np.ndarray | torch.Tensor, actual_classes: np.ndarray | torch.Tensor) -> None:
        if isinstance(predicted_logits, np.ndarray):
            predicted_logits = torch.tensor(predicted_logits)
        if isinstance(actual_classes, np.ndarray):
            actual_classes = torch.tensor(actual_classes)

        if self.mode == "multi-label":
            predicted_classes = (predicted_logits > 0.5).int()
            if len(self.true_positives) == 0:
                self.total_labels = predicted_classes.shape[1]
                self.true_positives = [0] * self.total_labels
                self.false_positives = [0] * self.total_labels
                self.false_negatives = [0] * self.total_labels
                self.true_negatives = [0] * self.total_labels

            for label_index in range(self.total_labels):
                actual_positive = actual_classes[:, label_index] == 1
                actual_negative = actual_classes[:, label_index] == 0

                self.true_positives[label_index] += int((predicted_classes[:, label_index] & actual_positive).sum())
                self.false_positives[label_index] += int((predicted_classes[:, label_index] & actual_negative).sum())
                self.false_negatives[label_index] += int(
                    ((1 - predicted_classes[:, label_index]) & actual_positive).sum(),
                )
                self.true_negatives[label_index] += int(
                    ((1 - predicted_classes[:, label_index]) & actual_negative).sum(),
                )

        elif self.mode == "multi-class":
            predicted_classes = torch.argmax(predicted_logits, dim=1)
            actual_classes = actual_classes.int()

            self.total_samples += predicted_classes.shape[0]
            self.correct_predictions += int((predicted_classes == actual_classes).sum())

            if len(self.true_positives) == 0:
                self.total_labels = predicted_logits.shape[1]
                self.true_positives = [0] * self.total_labels
                self.false_positives = [0] * self.total_labels
                self.false_negatives = [0] * self.total_labels

            for label_index in range(self.total_labels):
                actual_positive = actual_classes == label_index
                predicted_positive = predicted_classes == label_index

                self.true_positives[label_index] += int((predicted_positive & actual_positive).sum())
                self.false_positives[label_index] += int((predicted_positive & (actual_classes != label_index)).sum())
                self.false_negatives[label_index] += int(
                    ((actual_classes == label_index) & (predicted_classes != label_index)).sum(),
                )

        elif self.mode == "binary":
            predicted_classes = torch.argmax(predicted_logits, dim=1)
            actual_classes = actual_classes.int()

            self.total_samples += predicted_classes.shape[0]

            actual_positive = actual_classes == 1
            actual_negative = actual_classes == 0

            self.true_positives = int((predicted_classes & actual_positive).sum())
            self.false_positives = int((predicted_classes & actual_negative).sum())
            self.false_negatives = int(((1 - predicted_classes) & actual_positive).sum())
            self.true_negatives = int(((1 - predicted_classes) & actual_negative).sum())

    def precision(self):
        if self.mode == "multi-label" or self.mode == "multi-class":
            precisions = []
            for tp, fp in zip(self.true_positives, self.false_positives, strict=False):
                if tp + fp == 0:
                    precisions.append(0)
                else:
                    precisions.append(tp / (tp + fp))
            return np.mean(precisions) if precisions else 0
        elif self.mode == "binary":
            tp, fp = self.true_positives, self.false_positives
            return tp / (tp + fp) if tp + fp != 0 else 0

    def reset(self):
        self.true_positives = []
        self.false_positives = []
        self.false_negatives = []
        self.true_negatives = []
        self.total_labels = 0
        self.total_samples = 0
        self.correct_predictions = 0

    def recall(self):
        if self.mode == "multi-label" or self.mode == "multi-class":
            recalls = []
            for tp, fn in zip(self.true_positives, self.false_negatives, strict=False):
                if tp + fn == 0:
                    recalls.append(0)
                else:
                    recalls.append(tp / (tp + fn))
            return np.mean(recalls) if recalls else 0
        elif self.mode == "binary":
            tp, fn = self.true_positives, self.false_negatives
            return tp / (tp + fn) if tp + fn != 0 else 0

    def accuracy(self):
        if self.mode == "multi-label":
            accuracies = []
            for tp, tn, fp, fn in zip(
                self.true_positives,
                self.true_negatives,
                self.false_positives,
                self.false_negatives,
                strict=False,
            ):
                total = tp + tn + fp + fn
                if total == 0:
                    accuracies.append(0)
                else:
                    accuracies.append((tp + tn) / total)
            return np.mean(accuracies) if accuracies else 0
        elif self.mode == "multi-class":
            return self.correct_predictions / self.total_samples if self.total_samples != 0 else 0
        elif self.mode == "binary":
            tp, tn, fp, fn = self.true_positives, self.true_negatives, self.false_positives, self.false_negatives
            total = tp + tn + fp + fn
            return (tp + tn) / total if total != 0 else 0

    def f1_score(self):
        if self.mode == "multi-label" or self.mode == "multi-class":
            f1_scores = []
            for tp, fp, fn in zip(self.true_positives, self.false_positives, self.false_negatives, strict=False):
                precision = tp / (tp + fp) if tp + fp != 0 else 0
                recall = tp / (tp + fn) if tp + fn != 0 else 0
                if precision + recall == 0:
                    f1_scores.append(0)
                else:
                    f1_scores.append(2 * (precision * recall) / (precision + recall))
            return np.mean(f1_scores) if f1_scores else 0
        elif self.mode == "binary":
            tp, fp, fn = self.true_positives, self.false_positives, self.false_negatives
            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            return 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    def calculate_metrics(self) -> dict:
        return {
            "accuracy": self.accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1_score(),
        }
