"""LSTM classifiers over utterance embeddings."""

from __future__ import annotations

import keras
from keras import layers


def _metrics() -> list:
    """Stateful metric objects, which accumulate across batches."""
    return [
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ]


def build_single_head(
    vocab_size: int,
    sequence_length: int,
    num_classes: int,
    hidden_units: int = 100,
    embedding_dim: int = 100,
) -> keras.Model:
    """One LSTM, one softmax over the flat set of intents."""
    inputs = keras.Input(shape=(sequence_length,), name="tokens")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    x = layers.LSTM(hidden_units, dropout=0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="intent")(x)

    model = keras.Model(inputs, outputs, name=f"intent_lstm_{hidden_units}")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=_metrics())
    return model


def build_two_head(
    vocab_size: int,
    sequence_length: int,
    num_main_classes: int,
    num_sub_classes: int,
    hidden_units: int = 100,
    embedding_dim: int = 100,
) -> keras.Model:
    """Shared encoder, two softmax heads for the coarse and fine intent labels.

    Sharing the LSTM lets the fine-grained head borrow whatever the coarse task
    learns, which matters because the sub-class labels are far sparser.
    """
    inputs = keras.Input(shape=(sequence_length,), name="tokens")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    shared = layers.LSTM(hidden_units, dropout=0.2, name="shared_encoder")(x)

    main = layers.Dense(num_main_classes, activation="softmax", name="main_class")(shared)
    sub = layers.Dense(num_sub_classes, activation="softmax", name="sub_class")(shared)

    model = keras.Model(inputs, [main, sub], name=f"intent_two_head_{hidden_units}")
    model.compile(
        optimizer="adam",
        loss={"main_class": "categorical_crossentropy", "sub_class": "categorical_crossentropy"},
        metrics={"main_class": _metrics(), "sub_class": _metrics()},
    )
    return model
