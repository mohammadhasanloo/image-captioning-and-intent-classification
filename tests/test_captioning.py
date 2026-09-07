"""Tests for the captioner and its greedy decoder."""

from __future__ import annotations

import keras
import numpy as np

from captioning.decoding import greedy_decode
from captioning.model import build_captioner

VOCAB, MAX_LENGTH, IMAGE_DIM = 30, 8, 32


def tiny_backbone(image_dim: int) -> keras.Model:
    """Stand-in for ResNet, so tests need no pretrained download."""
    inputs = keras.Input(shape=(image_dim, image_dim, 3))
    x = keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inputs)
    return keras.Model(inputs, x, name="tiny_backbone")


def test_captioner_predicts_a_word_distribution_per_position():
    captioner, _ = build_captioner(
        VOCAB, MAX_LENGTH, image_dim=IMAGE_DIM, backbone=tiny_backbone
    )
    images = np.random.default_rng(0).random((2, IMAGE_DIM, IMAGE_DIM, 3)).astype("float32")
    tokens = np.zeros((2, MAX_LENGTH - 1), dtype=np.int32)
    logits = captioner.predict([images, tokens], verbose=0)

    # One extra position: the image is prepended to the token sequence.
    assert logits.shape == (2, MAX_LENGTH, VOCAB)
    assert np.allclose(logits.sum(axis=2), 1.0, atol=1e-4)


def test_encoder_is_returned_so_it_can_be_reused_across_decoding_steps():
    _, encoder = build_captioner(
        VOCAB, MAX_LENGTH, image_dim=IMAGE_DIM, backbone=tiny_backbone
    )
    images = np.random.default_rng(1).random((1, IMAGE_DIM, IMAGE_DIM, 3)).astype("float32")
    assert encoder.predict(images, verbose=0).shape == (1, 1, 300)


def test_freezing_flag_controls_backbone_trainability():
    frozen, _ = build_captioner(
        VOCAB, MAX_LENGTH, trainable_cnn=False, image_dim=IMAGE_DIM, backbone=tiny_backbone
    )
    unfrozen, _ = build_captioner(
        VOCAB, MAX_LENGTH, trainable_cnn=True, image_dim=IMAGE_DIM, backbone=tiny_backbone
    )
    assert frozen.count_params() > 0
    assert sum(w.size for w in unfrozen.get_weights()) == sum(
        w.size for w in frozen.get_weights()
    )
    # The fine-tuned variant has strictly more weights receiving gradients.
    assert len(unfrozen.trainable_weights) > len(frozen.trainable_weights)


def test_greedy_decode_stops_at_the_end_token():
    end_token = 7

    def step(tokens):
        probabilities = np.zeros((1, MAX_LENGTH, VOCAB))
        probabilities[0, :, end_token] = 1.0
        return probabilities

    assert greedy_decode(step, MAX_LENGTH, end_token) == [end_token]


def test_greedy_decode_runs_to_the_length_limit_without_an_end_token():
    def step(tokens):
        probabilities = np.zeros((1, MAX_LENGTH, VOCAB))
        probabilities[0, :, 3] = 1.0
        return probabilities

    assert greedy_decode(step, MAX_LENGTH, end_token=99) == [3] * (MAX_LENGTH - 1)


def test_greedy_decode_feeds_each_token_back_in():
    """Step n must see the tokens chosen at steps 0..n-1."""
    seen = []

    def step(tokens):
        seen.append(tokens.copy())
        probabilities = np.zeros((1, MAX_LENGTH, VOCAB))
        probabilities[0, :, 5] = 1.0
        return probabilities

    greedy_decode(step, MAX_LENGTH, end_token=99)
    assert seen[1][0, 0] == 5
    assert seen[2][0, 1] == 5
