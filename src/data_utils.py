"""
data_utils.py
-------------
Dataset loading, preprocessing, and augmentation utilities.

Supported datasets
------------------
- DIV2K     (training)
- NIH Chest X-ray
- INBreast
- Camelyon16 (patches)
- Kodak
- Custom folders
"""

import os
import glob
import numpy as np
import cv2
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Core image I/O
# ---------------------------------------------------------------------------

def load_image(path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Load a single image, resize, and normalise to [0, 1].

    Parameters
    ----------
    path        : str   path to image file
    target_size : (H, W)

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype float32, values in [0, 1]
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size[1], target_size[0]))
    return img.astype(np.float32) / 255.0


def save_image(image: np.ndarray, path: str):
    """Save a [0,1]-normalised image to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _image_paths(folder: str) -> List[str]:
    """Return all image paths in a folder (non-recursive)."""
    paths = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(paths)


def load_dataset(
    folder: str,
    target_size: Tuple[int, int] = (256, 256),
    max_images: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Load all images from a folder into a NumPy array.

    Parameters
    ----------
    folder      : str   directory containing images
    target_size : (H, W)
    max_images  : int or None  cap the number of images loaded
    verbose     : bool  print progress

    Returns
    -------
    np.ndarray  shape (N, H, W, 3), dtype float32
    """
    paths = _image_paths(folder)
    if max_images:
        paths = paths[:max_images]

    images = []
    for i, p in enumerate(paths):
        try:
            img = load_image(p, target_size)
            images.append(img)
            if verbose and (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(paths)} images …")
        except Exception as e:
            print(f"  [WARN] Skipping {p}: {e}")

    arr = np.array(images, dtype=np.float32)
    if verbose:
        print(f"  Dataset loaded: {arr.shape}  (N={len(arr)}, size={target_size})")
    return arr


def load_div2k(
    root: str = "data/DIV2K",
    split: str = "train",
    target_size: Tuple[int, int] = (256, 256),
    max_images: Optional[int] = None,
) -> np.ndarray:
    """Load DIV2K dataset split ('train' | 'valid' | 'test')."""
    folder = os.path.join(root, split)
    print(f"[DIV2K] Loading {split} set from: {folder}")
    return load_dataset(folder, target_size=target_size, max_images=max_images)


def load_nih_chest_xray(
    root: str = "data/NIH_Chest_Xray/images",
    target_size: Tuple[int, int] = (256, 256),
    max_images: Optional[int] = 500,
) -> np.ndarray:
    """Load NIH Chest X-ray images."""
    print(f"[NIH Chest X-ray] Loading from: {root}")
    return load_dataset(root, target_size=target_size, max_images=max_images)


def load_inbreast(
    root: str = "data/INBreast/images",
    target_size: Tuple[int, int] = (256, 256),
    max_images: Optional[int] = None,
) -> np.ndarray:
    """Load INBreast mammography dataset."""
    print(f"[INBreast] Loading from: {root}")
    return load_dataset(root, target_size=target_size, max_images=max_images)


def load_kodak(
    root: str = "data/Kodak",
    target_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Load the Kodak benchmark dataset (24 images)."""
    print(f"[Kodak] Loading from: {root}")
    return load_dataset(root, target_size=target_size)


def load_camelyon16_patches(
    root: str = "data/Camelyon16/patches",
    target_size: Tuple[int, int] = (256, 256),
    max_images: Optional[int] = 500,
) -> np.ndarray:
    """Load pre-extracted Camelyon16 patches."""
    print(f"[Camelyon16] Loading from: {root}")
    return load_dataset(root, target_size=target_size, max_images=max_images)


# ---------------------------------------------------------------------------
# Train / Test split
# ---------------------------------------------------------------------------

def train_test_split(
    data: np.ndarray,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly split dataset into train and test subsets."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    n_test = int(len(data) * test_ratio)
    return data[idx[n_test:]], data[idx[:n_test]]


# ---------------------------------------------------------------------------
# Augmentation (optional, for training robustness)
# ---------------------------------------------------------------------------

def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Apply random horizontal/vertical flip and 90° rotation.
    Input and output are float32 arrays in [0, 1].
    """
    ops = np.random.randint(0, 4)
    img = image.copy()
    if ops & 1:
        img = np.fliplr(img)
    if ops & 2:
        img = np.flipud(img)
    k = np.random.randint(0, 4)
    img = np.rot90(img, k=k)
    return img


def augment_dataset(data: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Augment a dataset by applying random transformations.

    Parameters
    ----------
    data   : (N, H, W, C)
    factor : int  how many augmented copies to add per original

    Returns
    -------
    np.ndarray  shape ((factor+1)*N, H, W, C)
    """
    augmented = [data]
    for _ in range(factor):
        aug = np.array([augment_image(img) for img in data])
        augmented.append(aug)
    return np.concatenate(augmented, axis=0)


# ---------------------------------------------------------------------------
# Compression-ratio helper
# ---------------------------------------------------------------------------

def estimate_compressed_bytes(bottleneck: np.ndarray, bits: int = 8) -> int:
    """
    Estimate bytes of a quantised bottleneck vector.

    Parameters
    ----------
    bottleneck : np.ndarray  latent vector
    bits       : int         quantisation bit-depth (default 8)

    Returns
    -------
    int  number of bytes
    """
    total_bits = bottleneck.size * bits
    return int(np.ceil(total_bits / 8))
