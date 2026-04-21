"""
metrics.py
----------
Evaluation metrics for image compression quality.

Metrics implemented
-------------------
- PSNR   – Peak Signal-to-Noise Ratio         (higher is better)
- SSIM   – Structural Similarity Index         (higher is better)
- MS-SSIM – Multi-Scale SSIM                   (higher is better)
- BPP    – Bits Per Pixel                      (lower = more compression)
- MSE    – Mean Squared Error                  (lower is better)
- PRD    – Percent Root-mean-square Difference (lower is better)
"""

import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio as _psnr,
    structural_similarity as _ssim,
    mean_squared_error as _mse,
)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Parameters
    ----------
    original, reconstructed : np.ndarray  values in [0, 1]

    Returns
    -------
    float  dB value; higher is better (≥ 40 dB is considered good)
    """
    orig = np.clip(original, 0, 1).astype(np.float64)
    rec  = np.clip(reconstructed, 0, 1).astype(np.float64)
    return float(_psnr(orig, rec, data_range=1.0))


def ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Structural Similarity Index.

    Returns
    -------
    float  in [0, 1]; higher is better
    """
    orig = np.clip(original, 0, 1).astype(np.float64)
    rec  = np.clip(reconstructed, 0, 1).astype(np.float64)
    channel_axis = 2 if orig.ndim == 3 else None
    return float(_ssim(orig, rec, data_range=1.0, channel_axis=channel_axis))


def ms_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Multi-Scale Structural Similarity Index.
    Uses TensorFlow's implementation for correctness.

    Returns
    -------
    float  in [0, 1]; higher is better
    """
    try:
        import tensorflow as tf
        orig = tf.constant(original[np.newaxis].astype(np.float32))
        rec  = tf.constant(reconstructed[np.newaxis].astype(np.float32))
        score = tf.image.ssim_multiscale(orig, rec, max_val=1.0)
        return float(score.numpy().mean())
    except Exception:
        # Fallback: return plain SSIM
        return ssim(original, reconstructed)


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Mean Squared Error.  Lower is better."""
    orig = np.clip(original, 0, 1).astype(np.float64)
    rec  = np.clip(reconstructed, 0, 1).astype(np.float64)
    return float(_mse(orig, rec))


def prd(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Percent Root-mean-square Difference (PRD).

        PRD = 100 * ||x - x̂||₂ / ||x||₂   (%)

    Lower is better.
    """
    orig = np.clip(original, 0, 1).astype(np.float64).flatten()
    rec  = np.clip(reconstructed, 0, 1).astype(np.float64).flatten()
    numerator   = np.sqrt(np.sum((orig - rec) ** 2))
    denominator = np.sqrt(np.sum(orig ** 2)) + 1e-10
    return float(100.0 * numerator / denominator)


def bpp(compressed_bytes: int, image_shape: tuple) -> float:
    """
    Bits Per Pixel.

    Parameters
    ----------
    compressed_bytes : int    size of compressed representation in bytes
    image_shape      : tuple  (H, W) or (H, W, C)

    Returns
    -------
    float  bits per pixel; lower means better compression
    """
    H, W = image_shape[0], image_shape[1]
    n_pixels = H * W
    return float((compressed_bytes * 8) / n_pixels)


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def evaluate_all(
    original: np.ndarray,
    reconstructed: np.ndarray,
    compressed_bytes: int = None,
) -> dict:
    """
    Compute all compression quality metrics.

    Parameters
    ----------
    original, reconstructed : np.ndarray  values in [0, 1]
    compressed_bytes        : int or None  if None, BPP is skipped

    Returns
    -------
    dict with keys: psnr, ssim, ms_ssim, mse, prd, bpp (optional)
    """
    results = {
        "psnr":    psnr(original, reconstructed),
        "ssim":    ssim(original, reconstructed),
        "ms_ssim": ms_ssim(original, reconstructed),
        "mse":     mse(original, reconstructed),
        "prd":     prd(original, reconstructed),
    }
    if compressed_bytes is not None:
        results["bpp"] = bpp(compressed_bytes, original.shape)
    return results


def print_metrics(metrics: dict, dataset_name: str = ""):
    """Pretty-print evaluation metrics."""
    header = f"  [{dataset_name}]" if dataset_name else ""
    print(f"\n{'─'*50}")
    print(f"  Evaluation Results{header}")
    print(f"{'─'*50}")
    print(f"  PSNR     : {metrics.get('psnr', 'N/A'):>8.2f} dB")
    print(f"  SSIM     : {metrics.get('ssim', 'N/A'):>8.4f}")
    print(f"  MS-SSIM  : {metrics.get('ms_ssim', 'N/A'):>8.4f}")
    print(f"  MSE      : {metrics.get('mse', 'N/A'):>12.6f}")
    print(f"  PRD      : {metrics.get('prd', 'N/A'):>8.2f} %")
    if "bpp" in metrics:
        print(f"  BPP      : {metrics['bpp']:>8.5f} bits/pixel")
    print(f"{'─'*50}\n")
