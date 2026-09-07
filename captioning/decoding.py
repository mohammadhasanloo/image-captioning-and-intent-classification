"""Turning a trained captioner into a caption."""

from __future__ import annotations

import numpy as np


def greedy_decode(
    decoder_step,
    max_length: int,
    end_token: int,
    start_tokens: np.ndarray | None = None,
) -> list[int]:
    """Generate tokens one at a time, taking the most likely word at each step.

    ``decoder_step`` takes the current token buffer and returns per-position word
    probabilities. Keeping it as a callable lets the caller decide what to cache
    between steps, so the image is encoded once rather than once per word.
    """
    tokens = np.zeros((1, max_length - 1), dtype=np.int32)
    if start_tokens is not None:
        length = min(len(start_tokens), max_length - 1)
        tokens[0, :length] = start_tokens[:length]

    generated: list[int] = []
    for position in range(max_length - 1):
        probabilities = decoder_step(tokens)
        token = int(np.argmax(probabilities[0, position]))
        tokens[0, position] = token
        generated.append(token)
        if token == end_token:
            break
    return generated


def caption_image(captioner, encoder, image: np.ndarray, tokenizer, max_length: int) -> str:
    """Caption a single image, encoding it once rather than once per word."""
    image = np.expand_dims(image, axis=0) if image.ndim == 3 else image
    end_token = tokenizer.texts_to_sequences(["endseq"])[0][0]

    def step(tokens: np.ndarray) -> np.ndarray:
        return captioner.predict([image, tokens], verbose=0)

    tokens = greedy_decode(step, max_length, end_token)
    words = tokenizer.sequences_to_texts([tokens])[0].split()
    return " ".join(w for w in words if w != "endseq")
