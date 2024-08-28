import numpy as np
import pytest
import torch

from efficient_multilingual_continual_pretraining.metrics import MetricCalculator


# Basic operations tests
@pytest.mark.parametrize(
    "input_data, expected",
    [
        (
            torch.tensor([1, 3, 5, 0]),
            torch.tensor(
                [
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                ]
            ),
        ),
        (
            torch.tensor([0, 0, 0, 1, 1, 1]),
            torch.tensor(
                [
                    [1, 0],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ]
            ),
        ),
    ],
)
def test_matrix_expansion(input_data, expected):
    assert (expected == MetricCalculator._expand_matrix(input_data, max(input_data) + 1)).all()


# Multiclass metric calculation testing
def test_multiclass_correct_update_perfect():
    predicted_logits = torch.tensor([[-10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]])
    true_classes = torch.tensor([2, 2, 2])
    calculator = MetricCalculator(torch.device("cpu"), "multi-class", 4)
    calculator.update(predicted_logits, true_classes)
    assert (calculator.true_positives == torch.tensor([0, 0, 3, 0])).all(), "Incorrect true positive count"
    assert (calculator.true_negatives == torch.tensor([3, 3, 0, 3])).all(), "Incorrect true negative count"
    assert (calculator.false_positives == torch.tensor([0, 0, 0, 0])).all(), "Incorrect false positive count"
    assert (calculator.false_negatives == torch.tensor([0, 0, 0, 0])).all(), "Incorrect false negative count"


def test_multiclass_correct_update_common():
    predicted_logits = torch.tensor([[10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]])
    true_classes = torch.tensor([0, 1, 2])
    calculator = MetricCalculator(torch.device("cpu"), "multi-class", 4)
    calculator.update(predicted_logits, true_classes)
    assert (calculator.true_positives == torch.tensor([1, 0, 1, 0])).all(), "Incorrect true positive count"
    assert (calculator.true_negatives == torch.tensor([2, 2, 1, 3])).all(), "Incorrect true negative count"
    assert (calculator.false_positives == torch.tensor([0, 0, 1, 0])).all(), "Incorrect false positive count"
    assert (calculator.false_negatives == torch.tensor([0, 1, 0, 0])).all(), "Incorrect false negative count"


@pytest.mark.parametrize(
    "predicted_logits, n_classes, true_classes",
    [
        (torch.tensor([[-10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]]), 4, torch.tensor([2, 2, 2])),
        (torch.tensor([[10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.1], [0.1, 0, -0.1, 0.05]]), 4, torch.tensor([0, 2, 0])),
    ],
)
def test_multiclass_correct(predicted_logits, n_classes, true_classes):
    calculator = MetricCalculator(torch.device("cpu"), "multi-class", n_classes)
    calculator.update(predicted_logits, true_classes)
    assert 1 == calculator._calculate_accuracy(), "Incorrect accuracy"
    assert 1 == calculator._calculate_precision(), "Incorrect precision"
    assert 1 == calculator._calculate_recall(), "Incorrect recall"
    assert 1 == calculator._calculate_f1(), "Incorrect f1 score"


def test_multiclass_common():
    predicted_logits = torch.tensor([[10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]])
    true_classes = torch.tensor([0, 1, 2])
    calculator = MetricCalculator(torch.device("cpu"), "multi-class", 4)
    calculator.update(predicted_logits, true_classes)
    assert np.allclose((1 + 2 / 3 + 2 / 3 + 1) / 4, calculator._calculate_accuracy()), "Incorrect accuracy"
    assert np.allclose((1 + 1 + 1 / 2 + 1) / 4, calculator._calculate_precision()), "Incorrect precision"
    assert np.allclose((1 + 0 + 1 + 1) / 4, calculator._calculate_recall()), "Incorrect recall"
    assert np.allclose((1 + 0 + 2 / 3 + 1) / 4, calculator._calculate_f1()), "Incorrect f1 score"


# Multilabel tests
def test_multilabel_correct_update_perfect():
    predicted_logits = torch.tensor([[-10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]])
    true_classes = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]])
    calculator = MetricCalculator(torch.device("cpu"), "multi-label", 4)
    calculator.update(predicted_logits, true_classes)
    assert (calculator.true_positives == torch.tensor([1, 3, 3, 3])).all(), "Incorrect true positive count"
    assert (calculator.true_negatives == torch.tensor([2, 0, 0, 0])).all(), "Incorrect true negative count"
    assert (calculator.false_positives == torch.tensor([0, 0, 0, 0])).all(), "Incorrect false positive count"
    assert (calculator.false_negatives == torch.tensor([0, 0, 0, 0])).all(), "Incorrect false negative count"


def test_multilabel_correct_update_common():
    predicted_logits = torch.tensor([[-10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]])
    true_classes = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1], [0, 0, 1, 1]])
    calculator = MetricCalculator(torch.device("cpu"), "multi-label", 4)
    calculator.update(predicted_logits, true_classes)
    assert (calculator.true_positives == torch.tensor([1, 1, 3, 2])).all(), "Incorrect true positive count"
    assert (calculator.true_negatives == torch.tensor([1, 0, 0, 0])).all(), "Incorrect true negative count"
    # 0 is converted to probability 0.5 which is counted towards positive.
    assert (calculator.false_positives == torch.tensor([0, 2, 0, 1])).all(), "Incorrect false positive count"
    assert (calculator.false_negatives == torch.tensor([1, 0, 0, 0])).all(), "Incorrect false negative count"


@pytest.mark.parametrize(
    "predicted_logits, n_classes, true_classes",
    [
        (
            torch.tensor([[-10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]]),
            4,
            torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]]),
        ),
        (
            torch.tensor([[10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.1], [0.1, 0, -0.1, 0.05]]),
            4,
            torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1]]),
        ),
    ],
)
def test_multilabel_correct(predicted_logits, n_classes, true_classes):
    calculator = MetricCalculator(torch.device("cpu"), "multi-label", n_classes)
    calculator.update(predicted_logits, true_classes)
    assert 1 == calculator._calculate_accuracy(), "Incorrect accuracy"
    assert 1 == calculator._calculate_precision(), "Incorrect precision"
    assert 1 == calculator._calculate_recall(), "Incorrect recall"
    assert 1 == calculator._calculate_f1(), "Incorrect f1 score"


def test_multilabel_common():
    predicted_logits = torch.tensor([[-10, 1, 5, 0.1], [0.1, 0.2, 0.3, 0.05], [-0.1, 0, 0.5, 0.1]])
    true_classes = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]])
    calculator = MetricCalculator(torch.device("cpu"), "multi-label", 4)
    calculator.update(predicted_logits, true_classes)
    assert np.allclose((1 / 3 + 1 / 3 + 1 + 1 / 3) / 4, calculator._calculate_accuracy()), "Incorrect accuracy"
    assert np.allclose((1 + 1 / 3 + 1 + 1 / 3) / 4, calculator._calculate_precision()), "Incorrect precision"
    assert np.allclose((1 / 3 + 1 + 1 + 1) / 4, calculator._calculate_recall()), "Incorrect recall"
    assert np.allclose(
        (2 * (1 / 3) / (4 / 3) + 2 * (1 / 3) / (4 / 3) + 1 + 2 * (1 / 3) / (4 / 3)) / 4, calculator._calculate_f1()
    ), "Incorrect f1 score"


#
# def test_multi_label_update_and_metrics(metric_calculators):
#     predicted_logits = torch.tensor([[0.6, 0.2], [0.3, 0.7]])
#     actual_classes = torch.tensor([[1, 0], [0, 1]])
#
#     calculator = metric_calculators["multi_label"]
#     calculator.update(predicted_logits, actual_classes)
#     metrics = calculator.calculate_metrics()
#
#     assert metrics["accuracy"] == 1.0
#     assert metrics["precision"] == 1.0
#     assert metrics["recall"] == 1.0
#     assert metrics["f1"] == 1.0
#
#
# def test_multi_class_update_and_metrics(metric_calculators):
#     predicted_logits = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
#     actual_classes = torch.tensor([2, 1, 0])
#
#     calculator = metric_calculators["multi_class"]
#     calculator.update(predicted_logits, actual_classes)
#     metrics = calculator.calculate_metrics()
#
#     assert metrics["accuracy"] == 1.0
#     assert metrics["precision"] == 1.0
#     assert metrics["recall"] == 1.0
#     assert metrics["f1"] == 1.0
#
#
# def test_binary_update_and_metrics(metric_calculators):
#     predicted_logits = torch.tensor([[0.8, 0.9], [0.4, 0.11], [0.6, 0.8], [0.1, 0.001]])
#     actual_classes = torch.tensor([1, 0, 1, 0])
#
#     calculator = metric_calculators["binary"]
#     calculator.update(predicted_logits, actual_classes)
#     metrics = calculator.calculate_metrics()
#
#     assert metrics["accuracy"] == 1.0
#     assert metrics["precision"] == 1.0
#     assert metrics["recall"] == 1.0
#     assert metrics["f1"] == 1.0
#
#
# def test_multi_label_partial_accuracy(metric_calculators):
#     predicted_logits = torch.tensor([[0.6, 0.2], [0.3, 0.7], [0.4, 0.6]])
#     actual_classes = torch.tensor([[1, 0], [0, 1], [1, 0]])
#
#     calculator = metric_calculators["multi_label"]
#     calculator.update(predicted_logits, actual_classes)
#     metrics = calculator.calculate_metrics()
#
#     assert metrics["accuracy"] == pytest.approx(2 / 3)
#     assert metrics["precision"] == pytest.approx(1.0)
#     assert metrics["recall"] == pytest.approx(2 / 3)
#     assert metrics["f1"] == pytest.approx(0.8, 0.01)
#
#
# def test_multi_class_partial_accuracy(metric_calculators):
#     predicted_logits = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.8, 0.1], [0.6, 0.2, 0.2]])
#     actual_classes = torch.tensor([2, 0, 0])
#
#     calculator = metric_calculators["multi_class"]
#     calculator.update(predicted_logits, actual_classes)
#     metrics = calculator.calculate_metrics()
#
#     assert metrics["accuracy"] == pytest.approx(2 / 3)
#     assert metrics["precision"] == pytest.approx(2 / 3)
#     assert metrics["recall"] == pytest.approx(2 / 3)
#     assert metrics["f1"] == pytest.approx(2 / 3)
#
#
# def test_binary_partial_accuracy(metric_calculators):
#     predicted_logits = torch.tensor([[0.8], [0.4], [0.6], [0.3]])
#     actual_classes = torch.tensor([1, 0, 1, 1])
#
#     calculator = metric_calculators["binary"]
#     calculator.update(predicted_logits, actual_classes)
#     metrics = calculator.calculate_metrics()
#
#     assert metrics["accuracy"] == pytest.approx(0.75)
#     assert metrics["precision"] == pytest.approx(2 / 3)
#     assert metrics["recall"] == pytest.approx(1.0)
#     assert metrics["f1"] == pytest.approx(0.8, 0.01)
