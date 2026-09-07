"""Show-and-Tell captioner: a CNN encoder feeding an LSTM decoder."""

from __future__ import annotations

from typing import Callable

import keras
from keras import layers

IMAGE_DIM = 224
FEATURE_DIM = 300
LSTM_UNITS = 256


def default_backbone(image_dim: int = IMAGE_DIM) -> keras.Model:
    """ResNet50 without its classification head.

    Chosen because it ships with Keras, so the package runs from a plain install.
    Pass a ``backbone`` to ``build_captioner`` to substitute another encoder.
    """
    return keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(image_dim, image_dim, 3)
    )


def build_captioner(
    vocab_size: int,
    max_length: int,
    trainable_cnn: bool = False,
    image_dim: int = IMAGE_DIM,
    backbone: keras.Model | Callable[[int], keras.Model] | None = None,
) -> tuple[keras.Model, keras.Model]:
    """Return (captioner, encoder).

    The image is projected to a single vector and prepended to the word embedding
    sequence, so the LSTM reads the picture as its first token and every word
    after that.

    The encoder is returned separately on purpose. Greedy decoding runs the
    decoder once per generated word, and without a handle on the encoder the
    whole CNN is re-run on the same image every single step.

    ``trainable_cnn`` selects the training regime. Leaving the encoder frozen is
    the setting that produces varied captions on this dataset; see the README.
    """
    if backbone is None:
        backbone = default_backbone(image_dim)
    elif callable(backbone) and not isinstance(backbone, keras.Model):
        backbone = backbone(image_dim)
    backbone.trainable = trainable_cnn

    image_input = keras.Input(shape=(image_dim, image_dim, 3), name="image")
    x = backbone(image_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(FEATURE_DIM, activation=layers.LeakyReLU(negative_slope=0.05))(x)
    features = layers.Reshape((1, FEATURE_DIM), name="image_features")(x)
    encoder = keras.Model(image_input, features, name="encoder")

    tokens_input = keras.Input(shape=(max_length - 1,), name="tokens")
    embeddings = layers.Embedding(vocab_size, FEATURE_DIM)(tokens_input)

    sequence = layers.Concatenate(axis=1)([features, embeddings])
    sequence = layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2)(sequence)
    logits = layers.Dense(vocab_size, activation="softmax", name="word")(sequence)

    captioner = keras.Model([image_input, tokens_input], logits, name="captioner")
    captioner.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return captioner, encoder
