"""
train.py
--------
Training entry-point for the Hybrid Image Compression model.

Usage
-----
    python train.py --data_dir data/DIV2K/train \
                    --epochs 50 \
                    --batch_size 16 \
                    --bottleneck 64 \
                    --output_dir outputs/

Arguments
---------
  --data_dir     : Path to training image folder
  --epochs       : Number of training epochs          (default 50)
  --batch_size   : Batch size                          (default 16)
  --bottleneck   : Bottleneck latent dimension          (default 64)
  --n_clusters   : K-Means cluster count               (default 4)
  --lr           : Learning rate                       (default 0.001)
  --output_dir   : Where to save weights + logs        (default outputs/)
  --max_images   : Cap on images loaded (for quick tests)
  --augment      : Whether to apply data augmentation  (flag)
"""

import argparse
import os
import json
import time
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")                 # headless backend
import matplotlib.pyplot as plt

from src.model import HybridCompressor, combined_loss
from src.data_utils import load_dataset, train_test_split, augment_dataset
from src.metrics import evaluate_all, print_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid Image Compression Model")
    p.add_argument("--data_dir",    type=str,   default="data/DIV2K/train")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--bottleneck",  type=int,   default=64)
    p.add_argument("--n_clusters",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--output_dir",  type=str,   default="outputs")
    p.add_argument("--max_images",  type=int,   default=None)
    p.add_argument("--augment",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def build_callbacks(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_weights.h5"),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv"),
            append=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_history(history, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss (MSE + SSIM)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    if "mse" in history.history:
        axes[1].plot(history.history["mse"],     label="Train MSE")
        axes[1].plot(history.history["val_mse"], label="Val MSE")
        axes[1].set_title("MSE")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[Train] Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Hybrid Deep Learning Image Compression – Training")
    print("=" * 60)
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Bottleneck  : {args.bottleneck}")
    print(f"  Clusters    : {args.n_clusters}")
    print(f"  Output dir  : {args.output_dir}")
    print("=" * 60 + "\n")

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("[Train] Loading dataset …")
    data = load_dataset(args.data_dir, max_images=args.max_images)
    if args.augment:
        print("[Train] Applying data augmentation …")
        data = augment_dataset(data, factor=2)

    X_train, X_val = train_test_split(data, test_ratio=0.1)
    print(f"[Train] Train: {X_train.shape}  |  Val: {X_val.shape}")

    # ── 2. Build & compile model ───────────────────────────────────────────
    print("[Train] Building model …")
    compressor = HybridCompressor(
        input_shape=(256, 256, 3),
        bottleneck_dim=args.bottleneck,
        n_clusters=args.n_clusters,
    )
    compressor.build()
    # Override LR
    tf.keras.backend.set_value(
        compressor.autoencoder.optimizer.lr, args.lr
    )
    compressor.autoencoder.summary()

    # ── 3. Fit K-Means on training data ────────────────────────────────────
    print("[Train] Fitting K-Means on GLCM features …")
    from src.model import extract_patch_glcm_features, cluster_regions
    sample_feats = []
    for img in X_train[:200]:
        feats, _ = extract_patch_glcm_features(img, patch_size=32)
        sample_feats.append(feats)
    all_feats = np.vstack(sample_feats)
    _, compressor.kmeans = cluster_regions(all_feats, n_clusters=args.n_clusters)
    print(f"[Train] K-Means fitted — {args.n_clusters} clusters.")

    # ── 4. Train ───────────────────────────────────────────────────────────
    callbacks = build_callbacks(args.output_dir)
    t0 = time.time()
    history = compressor.autoencoder.fit(
        X_train, X_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n[Train] Training completed in {elapsed:.1f}s")

    # ── 5. Save weights ────────────────────────────────────────────────────
    weights_path = os.path.join(args.output_dir, "final_weights.h5")
    compressor.save(weights_path)

    # ── 6. Quick validation metrics ───────────────────────────────────────
    print("[Train] Computing validation metrics on 10 samples …")
    sample_indices = np.random.choice(len(X_val), size=min(10, len(X_val)), replace=False)
    metric_list = []
    for i in sample_indices:
        orig = X_val[i]
        recon = compressor.autoencoder.predict(orig[np.newaxis], verbose=0)[0]
        recon = np.clip(recon, 0, 1)
        m = evaluate_all(orig, recon)
        metric_list.append(m)

    avg_metrics = {k: float(np.mean([m[k] for m in metric_list])) for k in metric_list[0]}
    print_metrics(avg_metrics, dataset_name="Validation Sample")

    # ── 7. Persist config + metrics ────────────────────────────────────────
    config = {
        "data_dir":    args.data_dir,
        "epochs":      args.epochs,
        "batch_size":  args.batch_size,
        "bottleneck":  args.bottleneck,
        "n_clusters":  args.n_clusters,
        "lr":          args.lr,
        "augment":     args.augment,
        "training_time_s": round(elapsed, 2),
        "val_metrics": avg_metrics,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Train] Config saved → {os.path.join(args.output_dir, 'config.json')}")

    # ── 8. Plot curves ─────────────────────────────────────────────────────
    plot_history(history, args.output_dir)

    print("\n[Train] ✅  All done.\n")


if __name__ == "__main__":
    main()
