import pytest
import torch

from efficient_multilingual_continual_pretraining.metrics import MetricCalculator


@pytest.fixture()
def metric_calculators():
    return {
        "multi_label": MetricCalculator(mode="multi-label"),
        "multi_class": MetricCalculator(mode="multi-class"),
        "binary": MetricCalculator(mode="binary"),
    }


def test_multi_label_update_and_metrics(metric_calculators):
    predicted_logits = torch.tensor([[0.6, 0.2], [0.3, 0.7]])
    actual_classes = torch.tensor([[1, 0], [0, 1]])

    calculator = metric_calculators["multi_label"]
    calculator.update(predicted_logits, actual_classes)
    metrics = calculator.calculate_metrics()

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_multi_class_update_and_metrics(metric_calculators):
    predicted_logits = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    actual_classes = torch.tensor([2, 1, 0])

    calculator = metric_calculators["multi_class"]
    calculator.update(predicted_logits, actual_classes)
    metrics = calculator.calculate_metrics()

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_binary_update_and_metrics(metric_calculators):
    predicted_logits = torch.tensor([[0.8, 0.9], [0.4, 0.11], [0.6, 0.8], [0.1, 0.001]])
    actual_classes = torch.tensor([1, 0, 1, 0])

    calculator = metric_calculators["binary"]
    calculator.update(predicted_logits, actual_classes)
    metrics = calculator.calculate_metrics()

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_multi_label_partial_accuracy(metric_calculators):
    predicted_logits = torch.tensor([[0.6, 0.2], [0.3, 0.7], [0.4, 0.6]])
    actual_classes = torch.tensor([[1, 0], [0, 1], [1, 0]])

    calculator = metric_calculators["multi_label"]
    calculator.update(predicted_logits, actual_classes)
    metrics = calculator.calculate_metrics()

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.8, 0.01)


def test_multi_class_partial_accuracy(metric_calculators):
    predicted_logits = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
    actual_classes = torch.tensor([2, 0, 0])

    calculator = metric_calculators["multi_class"]
    calculator.update(predicted_logits, actual_classes)
    metrics = calculator.calculate_metrics()

    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_binary_partial_accuracy(metric_calculators):
    predicted_logits = torch.tensor([[0.8], [0.4], [0.6], [0.3]])
    actual_classes = torch.tensor([1, 0, 1, 1])

    calculator = metric_calculators["binary"]
    calculator.update(predicted_logits, actual_classes)
    metrics = calculator.calculate_metrics()

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8, 0.01)
