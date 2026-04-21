"""
demo.py
-------
Interactive demo: compress a single image and display results.

Usage
-----
    python demo.py --image samples/chest_xray.png \
                   --weights outputs/best_weights.h5

    # With no weights (untrained model for architecture demo):
    python demo.py --image samples/test.jpg --no_weights
"""

import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import HybridCompressor, build_sdae
from src.data_utils import load_image, save_image, estimate_compressed_bytes
from src.metrics import evaluate_all, print_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Compression Demo")
    p.add_argument("--image",       type=str,  required=True,  help="Path to input image")
    p.add_argument("--weights",     type=str,  default=None,   help="Path to trained weights (.h5)")
    p.add_argument("--bottleneck",  type=int,  default=64)
    p.add_argument("--output_dir",  type=str,  default="results/demo")
    p.add_argument("--no_weights",  action="store_true", help="Run without pre-trained weights")
    return p.parse_args()


def show_glcm_features(image: np.ndarray, save_path: str):
    """Visualise GLCM texture map over the image."""
    from src.model import extract_patch_glcm_features
    feats, coords = extract_patch_glcm_features(image, patch_size=32)

    H, W = image.shape[:2]
    texture_map = np.zeros((H, W), dtype=np.float32)
    patch_size = 32

    # Use 'energy' (index 3 per channel, first channel)
    energy_idx = 3
    for feat_vec, (r, c) in zip(feats, coords):
        energy = feat_vec[energy_idx]
        texture_map[r : r + patch_size, c : c + patch_size] = energy

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(texture_map, cmap="viridis")
    axes[1].set_title("GLCM Energy Map (per patch)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Demo] GLCM map saved → {save_path}")


def show_cluster_map(image: np.ndarray, save_path: str, n_clusters: int = 4):
    """Visualise K-Means region clusters."""
    from src.model import extract_patch_glcm_features, cluster_regions

    feats, coords = extract_patch_glcm_features(image, patch_size=32)
    labels, _ = cluster_regions(feats, n_clusters=n_clusters)

    H, W = image.shape[:2]
    cluster_map = np.zeros((H, W), dtype=np.float32)
    patch_size = 32
    for label, (r, c) in zip(labels, coords):
        cluster_map[r : r + patch_size, c : c + patch_size] = label

    cmap = plt.cm.get_cmap("tab10", n_clusters)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(cluster_map, cmap=cmap, vmin=0, vmax=n_clusters - 1)
    axes[1].set_title(f"K-Means Clusters (k={n_clusters})")
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], ticks=range(n_clusters), fraction=0.046, pad=0.04)
    cbar.set_ticklabels([f"C{i}" for i in range(n_clusters)])

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Demo] Cluster map saved → {save_path}")


def show_swt_subbands(image: np.ndarray, save_path: str):
    """Visualise SWT approximation and detail sub-bands (first channel)."""
    from src.model import apply_swt

    coeffs_list = apply_swt(image, wavelet="haar", level=1)
    # Use first channel
    cA, (cH, cV, cD) = coeffs_list[0][0]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].imshow(np.clip(image, 0, 1))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    def show_band(ax, band, title):
        band_norm = (band - band.min()) / (band.ptp() + 1e-8)
        ax.imshow(band_norm, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    show_band(axes[0, 1], cA, "SWT — Approximation (LL)")
    show_band(axes[0, 2], cH, "SWT — Horizontal Detail (LH)")
    show_band(axes[1, 0], cV, "SWT — Vertical Detail (HL)")
    show_band(axes[1, 1], cD, "SWT — Diagonal Detail (HH)")
    axes[1, 2].axis("off")

    plt.suptitle("Stationary Wavelet Transform Sub-bands (Channel 0)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Demo] SWT sub-bands saved → {save_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Hybrid Compression Demo")
    print("=" * 60)

    # ── Load image ─────────────────────────────────────────────────────────
    print(f"\n[Demo] Loading image: {args.image}")
    image = load_image(args.image, target_size=(256, 256))
    print(f"[Demo] Image shape: {image.shape}  dtype: {image.dtype}")

    # ── Visualise intermediate stages ──────────────────────────────────────
    print("\n[Demo] Generating visualisations …")
    show_glcm_features(image, os.path.join(args.output_dir, "glcm_map.png"))
    show_cluster_map(image, os.path.join(args.output_dir, "cluster_map.png"))
    show_swt_subbands(image, os.path.join(args.output_dir, "swt_subbands.png"))

    # ── Build compressor ───────────────────────────────────────────────────
    compressor = HybridCompressor(
        input_shape=(256, 256, 3),
        bottleneck_dim=args.bottleneck,
        n_clusters=4,
    )
    compressor.build()

    if not args.no_weights and args.weights:
        compressor.load(args.weights)
    else:
        print("[Demo] Running without pre-trained weights (random init). Metrics will be poor.")

    # ── Compress / Decompress ──────────────────────────────────────────────
    print("\n[Demo] Compressing …")
    t0 = time.time()
    compressed = compressor.compress(image)
    t_compress = time.time() - t0

    print("[Demo] Decompressing …")
    t1 = time.time()
    reconstructed = compressor.decompress(compressed)
    t_decompress = time.time() - t1

    total_time = t_compress + t_decompress
    print(f"[Demo] Encode time : {t_compress:.4f}s")
    print(f"[Demo] Decode time : {t_decompress:.4f}s")
    print(f"[Demo] Total time  : {total_time:.4f}s")

    # ── Metrics ────────────────────────────────────────────────────────────
    cbytes = estimate_compressed_bytes(compressed["bottleneck"])
    metrics = evaluate_all(image, reconstructed, compressed_bytes=cbytes)
    metrics["encode_decode_time_s"] = round(total_time, 4)
    print_metrics(metrics, dataset_name=os.path.basename(args.image))

    # ── Save outputs ───────────────────────────────────────────────────────
    save_image(image,         os.path.join(args.output_dir, "original.png"))
    save_image(reconstructed, os.path.join(args.output_dir, "reconstructed.png"))

    # Full side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(image, 0, 1));          axes[0].set_title("Original");       axes[0].axis("off")
    axes[1].imshow(np.clip(reconstructed, 0, 1));  axes[1].set_title("Reconstructed"); axes[1].axis("off")
    diff = np.abs(image - reconstructed)
    axes[2].imshow(diff.mean(axis=-1), cmap="hot"); axes[2].set_title("Diff (abs)");   axes[2].axis("off")

    caption = (
        f"PSNR={metrics['psnr']:.2f} dB  |  SSIM={metrics['ssim']:.4f}  |  "
        f"PRD={metrics['prd']:.2f}%  |  Time={metrics['encode_decode_time_s']:.4f}s"
    )
    fig.suptitle(caption, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "full_comparison.png"), dpi=130)
    plt.close()

    print(f"\n[Demo] All outputs saved to: {args.output_dir}")
    print("[Demo] ✅  Demo complete.\n")


if __name__ == "__main__":
    main()
