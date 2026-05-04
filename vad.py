"""Silero VAD v6 (MLX) — vendored 256 ms unified-mode forward pass.

Ported from `example_256ms.py` in mlx-community/silero-vad-v6
(https://huggingface.co/mlx-community/silero-vad-v6).

Upstream attribution:
- snakers4/silero-vad — original architecture and weights, MIT licensed
  (https://github.com/snakers4/silero-vad)
- mlx-community/silero-vad-v6 — MLX conversion and reference scripts,
  MIT licensed

Weights are downloaded at runtime via huggingface_hub; nothing is
redistributed. Public API: load_vad(local_dir) and vad_probabilities(model, audio).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

CHUNK_SAMPLES = 512
BLOCK_TOTAL = 4096  # 8 chunks × 512 samples = 256 ms at 16 kHz
CONTEXT_SAMPLES = 64
STFT_HOP = 128
STFT_PAD_RIGHT = 64
LSTM_HIDDEN = 128
SAMPLE_RATE = 16_000

KEYS: tuple[str, ...] = (
    "vad_16k.stft_conv.weight",
    "vad_16k.conv1.weight",
    "vad_16k.conv1.bias",
    "vad_16k.conv2.weight",
    "vad_16k.conv2.bias",
    "vad_16k.conv3.weight",
    "vad_16k.conv3.bias",
    "vad_16k.conv4.weight",
    "vad_16k.conv4.bias",
    "vad_16k.lstm.Wx",
    "vad_16k.lstm.Wh",
    "vad_16k.lstm.bias",
    "vad_16k.final_conv.weight",
    "vad_16k.final_conv.bias",
)


def load_vad(local_dir: Path) -> dict[str, mx.array]:
    """Load weights from local_dir/model.safetensors as an MLX weight dict."""
    weights_path = Path(local_dir) / "model.safetensors"
    raw = load_file(str(weights_path))
    return {k: mx.array(raw[k]) for k in KEYS}


def _reflect_pad_right_576(samples_576: np.ndarray) -> np.ndarray:
    """PyTorch ReflectionPad1d(right=64): out[L+i] = in[L-2-i] for i in 0..63."""
    L = samples_576.shape[-1]
    pad = np.empty(STFT_PAD_RIGHT, dtype=samples_576.dtype)
    for i in range(STFT_PAD_RIGHT):
        pad[i] = samples_576[L - 2 - i]
    return np.concatenate([samples_576, pad], axis=-1)


def _predict_chunk(
    audio_640: mx.array, h: mx.array, c: mx.array, w: dict[str, mx.array]
) -> tuple[mx.array, mx.array, mx.array]:
    """Forward pass for one 32 ms chunk. Returns (prob, h_new, c_new)."""
    z = mx.conv1d(audio_640, w["vad_16k.stft_conv.weight"], stride=STFT_HOP, padding=0)
    real, imag = z[:, :, :129], z[:, :, 129:]
    x = mx.sqrt(real * real + imag * imag + 1e-12)
    cfgs = [(1, 1), (2, 1), (2, 1), (1, 1)]
    for i, (stride, padding) in enumerate(cfgs, start=1):
        x = mx.conv1d(x, w[f"vad_16k.conv{i}.weight"], stride=stride, padding=padding)
        x = x + w[f"vad_16k.conv{i}.bias"]
        x = mx.maximum(x, 0)
    feat = x[:, 0, :]
    gates = (
        feat @ w["vad_16k.lstm.Wx"].T
        + h @ w["vad_16k.lstm.Wh"].T
        + w["vad_16k.lstm.bias"]
    )
    i_g, f_g, g_g, o_g = mx.split(gates, 4, axis=-1)
    c_new = mx.sigmoid(f_g) * c + mx.sigmoid(i_g) * mx.tanh(g_g)
    h_new = mx.sigmoid(o_g) * mx.tanh(c_new)
    dec = mx.conv1d(
        mx.maximum(h_new, 0)[:, None, :],
        w["vad_16k.final_conv.weight"],
        stride=1,
        padding=0,
    )
    dec = dec + w["vad_16k.final_conv.bias"]
    return mx.sigmoid(dec), h_new, c_new


def _predict_block_256ms(
    block_audio_4096: np.ndarray,
    initial_ctx: np.ndarray,
    h: mx.array,
    c: mx.array,
    w: dict[str, mx.array],
) -> tuple[float, np.ndarray, mx.array, mx.array]:
    """Process one 256 ms block (8 × 32 ms chunks). LSTM state shared across
    the inner chunks; aggregate sub-probabilities via noisy-OR."""
    rolling_ctx = initial_ctx
    chunk_probs: list[mx.array] = []
    for ci in range(8):
        src = block_audio_4096[ci * CHUNK_SAMPLES : (ci + 1) * CHUNK_SAMPLES]
        merged = np.concatenate([rolling_ctx, src])
        padded = _reflect_pad_right_576(merged)
        audio = mx.array(padded[None, :, None])
        prob, h, c = _predict_chunk(audio, h, c, w)
        chunk_probs.append(prob)
        rolling_ctx = src[-CONTEXT_SAMPLES:].copy()

    product = 1.0 - chunk_probs[0]
    for ci in range(1, 8):
        product = product * (1.0 - chunk_probs[ci])
    agg = 1.0 - product
    mx.eval(agg, h, c)
    new_ctx = block_audio_4096[-CONTEXT_SAMPLES:].copy()
    return float(agg[0, 0, 0].item()), new_ctx, h, c


def vad_probabilities(model: dict[str, mx.array], audio: np.ndarray) -> np.ndarray:
    """Run 256 ms unified mode end-to-end over a 16 kHz mono float32 array.

    Returns a 1-D ndarray of probabilities, one per 256 ms window.
    LSTM state and rolling 64-sample context are carried across windows.
    Trailing samples are zero-padded to the next 256 ms boundary.
    Audio shorter than one full block produces a single probability after
    zero-padding; truly empty audio (0 samples) returns an empty array.
    """
    samples = np.ascontiguousarray(audio, dtype=np.float32)
    if samples.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)

    leftover = samples.shape[0] % BLOCK_TOTAL
    if leftover:
        pad = np.zeros(BLOCK_TOTAL - leftover, dtype=np.float32)
        samples = np.concatenate([samples, pad])
    n_blocks = samples.shape[0] // BLOCK_TOTAL

    h = mx.zeros((1, LSTM_HIDDEN))
    c = mx.zeros((1, LSTM_HIDDEN))
    initial_ctx = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    probs = np.zeros(n_blocks, dtype=np.float32)
    for bi in range(n_blocks):
        block_audio = samples[bi * BLOCK_TOTAL : (bi + 1) * BLOCK_TOTAL]
        prob, initial_ctx, h, c = _predict_block_256ms(
            block_audio, initial_ctx, h, c, model
        )
        probs[bi] = prob
    return probs
