"""
evaluate.py
-----------
Evaluate the trained Hybrid Compression model on a dataset or single image.

Usage
-----
# Evaluate on a folder of images:
    python evaluate.py --data_dir data/Kodak \
                       --weights  outputs/best_weights.h5 \
                       --output_dir results/kodak/

# Evaluate a single image and save reconstructed output:
    python evaluate.py --image path/to/image.png \
                       --weights outputs/best_weights.h5 \
                       --output_dir results/
"""

import argparse
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model import HybridCompressor
from src.data_utils import load_image, load_dataset, save_image, estimate_compressed_bytes
from src.metrics import evaluate_all, print_metrics, bpp as compute_bpp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Hybrid Image Compression Model")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--data_dir", type=str, help="Folder of test images")
    group.add_argument("--image",    type=str, help="Single image path")

    p.add_argument("--weights",     type=str, default="outputs/best_weights.h5")
    p.add_argument("--bottleneck",  type=int, default=64)
    p.add_argument("--n_clusters",  type=int, default=4)
    p.add_argument("--output_dir",  type=str, default="results")
    p.add_argument("--save_images", action="store_true", help="Save original + reconstructed pairs")
    p.add_argument("--max_images",  type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_comparison(original, reconstructed, metrics, save_path):
    """Save side-by-side original vs reconstructed figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(np.clip(original, 0, 1))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(np.clip(reconstructed, 0, 1))
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    diff = np.abs(original - reconstructed)
    axes[2].imshow(diff.mean(axis=-1) if diff.ndim == 3 else diff, cmap="hot")
    axes[2].set_title("Difference (abs)")
    axes[2].axis("off")

    info = (
        f"PSNR={metrics['psnr']:.2f} dB  |  "
        f"SSIM={metrics['ssim']:.4f}  |  "
        f"PRD={metrics['prd']:.2f}%"
    )
    fig.suptitle(info, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_metrics_bar(all_metrics: list, output_path: str):
    """Bar chart of PSNR and SSIM across images."""
    psnrs  = [m["psnr"]  for m in all_metrics]
    ssims  = [m["ssim"]  for m in all_metrics]
    prds   = [m["prd"]   for m in all_metrics]
    labels = [f"img_{i+1}" for i in range(len(all_metrics))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].bar(labels, psnrs, color="steelblue")
    axes[0].axhline(np.mean(psnrs), color="red", linestyle="--", label=f"Mean={np.mean(psnrs):.2f}")
    axes[0].set_title("PSNR (dB)")
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].legend()
    axes[0].grid(axis="y")

    axes[1].bar(labels, ssims, color="seagreen")
    axes[1].axhline(np.mean(ssims), color="red", linestyle="--", label=f"Mean={np.mean(ssims):.4f}")
    axes[1].set_title("SSIM")
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[1].legend()
    axes[1].grid(axis="y")

    axes[2].bar(labels, prds, color="tomato")
    axes[2].axhline(np.mean(prds), color="black", linestyle="--", label=f"Mean={np.mean(prds):.2f}%")
    axes[2].set_title("PRD (%)")
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[2].legend()
    axes[2].grid(axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[Evaluate] Metrics plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Core evaluation routine
# ---------------------------------------------------------------------------

def evaluate_single(compressor, image_path: str, output_dir: str, save_img: bool = True):
    """Compress and evaluate a single image."""
    orig_uint8 = None
    try:
        import cv2
        orig_uint8 = cv2.imread(image_path)
        orig_uint8 = cv2.cvtColor(orig_uint8, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    image = load_image(image_path)

    t0 = time.time()
    compressed = compressor.compress(image)
    reconstructed = compressor.decompress(compressed)
    elapsed = time.time() - t0

    # Estimate BPP from bottleneck size
    cbytes = estimate_compressed_bytes(compressed["bottleneck"])
    metrics = evaluate_all(image, reconstructed, compressed_bytes=cbytes)
    metrics["encode_decode_time_s"] = round(elapsed, 4)

    if save_img:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        save_image(reconstructed, os.path.join(output_dir, f"{stem}_reconstructed.png"))
        save_comparison(
            image, reconstructed, metrics,
            os.path.join(output_dir, f"{stem}_comparison.png"),
        )

    return metrics


def evaluate_dataset(compressor, data_dir: str, output_dir: str, save_imgs: bool, max_images=None):
    """Evaluate all images in a directory."""
    from src.data_utils import _image_paths
    paths = _image_paths(data_dir)
    if max_images:
        paths = paths[:max_images]

    all_metrics = []
    for i, p in enumerate(paths, 1):
        print(f"  [{i}/{len(paths)}] {os.path.basename(p)}", end=" … ")
        try:
            m = evaluate_single(compressor, p, output_dir, save_img=save_imgs)
            all_metrics.append(m)
            print(f"PSNR={m['psnr']:.2f} dB  SSIM={m['ssim']:.4f}  "
                  f"PRD={m['prd']:.2f}%  t={m['encode_decode_time_s']:.3f}s")
        except Exception as e:
            print(f"ERROR: {e}")

    return all_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Hybrid Deep Learning Image Compression – Evaluation")
    print("=" * 60)

    # ── Build & load model ─────────────────────────────────────────────────
    compressor = HybridCompressor(
        bottleneck_dim=args.bottleneck,
        n_clusters=args.n_clusters,
    )
    compressor.load(args.weights)

    # ── Run evaluation ─────────────────────────────────────────────────────
    if args.image:
        print(f"\n[Evaluate] Single image: {args.image}\n")
        metrics = evaluate_single(
            compressor, args.image, args.output_dir, save_img=True
        )
        print_metrics(metrics, dataset_name=os.path.basename(args.image))
        all_metrics = [metrics]
    else:
        print(f"\n[Evaluate] Dataset: {args.data_dir}\n")
        all_metrics = evaluate_dataset(
            compressor, args.data_dir, args.output_dir,
            save_imgs=args.save_images, max_images=args.max_images,
        )

    # ── Aggregate ──────────────────────────────────────────────────────────
    if all_metrics:
        avg = {k: float(np.mean([m[k] for m in all_metrics if k in m]))
               for k in all_metrics[0]}
        print_metrics(avg, dataset_name="AVERAGE")

        # Save JSON
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump({"average": avg, "per_image": all_metrics}, f, indent=2)
        print(f"[Evaluate] Results saved → {results_path}")

        # Bar plot (skip for single image)
        if len(all_metrics) > 1:
            plot_metrics_bar(
                all_metrics,
                os.path.join(args.output_dir, "metrics_bar.png"),
            )

    print("\n[Evaluate] ✅  Done.\n")


if __name__ == "__main__":
    main()
