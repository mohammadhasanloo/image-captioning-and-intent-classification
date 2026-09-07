"""Tests for the intent classifiers and their metrics."""

from __future__ import annotations

import numpy as np
import pytest

from intent.metrics import evaluate
from intent.model import build_single_head, build_two_head

VOCAB, LENGTH, MAIN, SUB = 50, 12, 4, 9


def test_metrics_are_perfect_when_predictions_match():
    labels = np.array([0, 1, 2, 1])
    result = evaluate(labels, labels)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.micro_f1 == 1.0


def test_metrics_accept_one_hot_and_integer_labels_alike():
    integers = np.array([0, 1, 2])
    one_hot = np.eye(3)[integers]
    assert evaluate(one_hot, one_hot).accuracy == evaluate(integers, integers).accuracy


def test_macro_f1_punishes_ignoring_a_rare_class():
    """High accuracy with a collapsed prediction is exactly what macro-F1 catches."""
    y_true = np.array([0] * 95 + [1] * 5)
    always_majority = np.zeros(100, dtype=int)
    result = evaluate(y_true, always_majority)

    assert result.accuracy == pytest.approx(0.95)
    assert result.macro_f1 < 0.5  # the rare class contributes nothing


def test_micro_and_macro_diverge_under_imbalance():
    y_true = np.array([0] * 90 + [1] * 10)
    predicted = np.array([0] * 95 + [1] * 5)
    result = evaluate(y_true, predicted)
    assert result.micro_f1 != pytest.approx(result.macro_f1)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), np.array([0]))


def test_single_head_maps_tokens_to_a_class_distribution():
    model = build_single_head(VOCAB, LENGTH, MAIN, hidden_units=16)
    tokens = np.random.default_rng(0).integers(0, VOCAB, (3, LENGTH))
    probabilities = model.predict(tokens, verbose=0)

    assert probabilities.shape == (3, MAIN)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_two_head_shares_one_encoder_between_both_outputs():
    model = build_two_head(VOCAB, LENGTH, MAIN, SUB, hidden_units=16)
    tokens = np.random.default_rng(1).integers(0, VOCAB, (2, LENGTH))
    main, sub = model.predict(tokens, verbose=0)

    assert main.shape == (2, MAIN)
    assert sub.shape == (2, SUB)
    encoders = [layer for layer in model.layers if layer.name == "shared_encoder"]
    assert len(encoders) == 1


def test_hidden_size_changes_parameter_count():
    small = build_single_head(VOCAB, LENGTH, MAIN, hidden_units=25)
    large = build_single_head(VOCAB, LENGTH, MAIN, hidden_units=100)
    assert large.count_params() > small.count_params()
