"""
model.py
--------
Hybrid Deep Learning Image Compression Model

Integrates:
  - GLCM (Gray-Level Co-occurrence Matrix) for texture feature extraction
  - K-Means clustering for region-based adaptive compression
  - SWT (Stationary Wavelet Transform) for multiresolution decomposition
  - SDAE (Stacked Denoising Autoencoder) for deep feature compression

Reference:
  Al-Khafaji & Ramaha (2025). Hybrid deep learning architecture for scalable
  and high-quality image compression. Scientific Reports 15, 22926.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import pywt
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# 1. GLCM Feature Extraction
# ---------------------------------------------------------------------------

def compute_glcm_features(image: np.ndarray) -> np.ndarray:
    """
    Compute GLCM texture features for each channel of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, C) with values in [0, 1].

    Returns
    -------
    features : np.ndarray
        Feature vector of length 5 * C (contrast, dissimilarity,
        homogeneity, energy, correlation) per channel.
    """
    from skimage.feature import graycomatrix, graycoprops

    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    feature_list = []
    for c in range(image.shape[2]):
        channel = (image[:, :, c] * 255).astype(np.uint8)
        glcm = graycomatrix(
            channel,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True,
        )
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
            feature_list.append(graycoprops(glcm, prop)[0, 0])

    return np.array(feature_list, dtype=np.float32)


def extract_patch_glcm_features(image: np.ndarray, patch_size: int = 32) -> np.ndarray:
    """
    Extract GLCM features patch-by-patch across the image.

    Returns
    -------
    patch_features : np.ndarray  shape (n_patches, n_features)
    patch_coords   : list of (row, col) top-left corners
    """
    H, W = image.shape[:2]
    features, coords = [], []

    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            patch = image[r : r + patch_size, c : c + patch_size]
            feat = compute_glcm_features(patch)
            features.append(feat)
            coords.append((r, c))

    return np.array(features, dtype=np.float32), coords


# ---------------------------------------------------------------------------
# 2. K-Means Region Clustering
# ---------------------------------------------------------------------------

def cluster_regions(features: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    """
    Apply K-Means to GLCM feature vectors.

    Returns
    -------
    labels   : np.ndarray  shape (n_patches,)
    kmeans   : fitted KMeans object
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    return labels, kmeans


# ---------------------------------------------------------------------------
# 3. Stationary Wavelet Transform (SWT)
# ---------------------------------------------------------------------------

def apply_swt(image: np.ndarray, wavelet: str = "haar", level: int = 1):
    """
    Apply 2-D Stationary Wavelet Transform to a single-channel image.

    Returns
    -------
    coeffs : list of (cA, (cH, cV, cD)) per level  (pywt.swt2 format)
    """
    if image.ndim == 3:
        # Process each channel independently
        all_coeffs = []
        for c in range(image.shape[2]):
            coeffs = pywt.swt2(image[:, :, c], wavelet=wavelet, level=level)
            all_coeffs.append(coeffs)
        return all_coeffs
    return pywt.swt2(image, wavelet=wavelet, level=level)


def apply_iswt(coeffs, wavelet: str = "haar") -> np.ndarray:
    """Inverse SWT — reconstruct from coefficients."""
    if isinstance(coeffs[0], list):
        # Multi-channel
        channels = [pywt.iswt2(ch_coeffs, wavelet=wavelet) for ch_coeffs in coeffs]
        return np.stack(channels, axis=-1)
    return pywt.iswt2(coeffs, wavelet=wavelet)


def coeffs_to_array(coeffs) -> np.ndarray:
    """Flatten SWT coefficients into a stacked numpy array."""
    arrays = []
    for cA, (cH, cV, cD) in coeffs:
        arrays.extend([cA, cH, cV, cD])
    return np.stack(arrays, axis=0)          # (4*level, H, W)


def array_to_coeffs(arr: np.ndarray, level: int = 1):
    """Restore pywt coefficient structure from stacked array."""
    coeffs = []
    for l in range(level):
        base = l * 4
        cA = arr[base]
        cH, cV, cD = arr[base + 1], arr[base + 2], arr[base + 3]
        coeffs.append((cA, (cH, cV, cD)))
    return coeffs


# ---------------------------------------------------------------------------
# 4. SDAE Architecture
# ---------------------------------------------------------------------------

def build_sdae(input_shape=(256, 256, 3), bottleneck_dim: int = 64) -> tuple:
    """
    Build the Stacked Denoising Autoencoder (SDAE).

    Architecture
    ------------
    Encoder  : Conv(32) → BN → MaxPool → Conv(16) → BN → MaxPool → Flatten → Dense(bottleneck)
    Decoder  : Dense → Reshape → UpSample → Conv(16) → BN → UpSample → Conv(32) → BN → Conv(3, sigmoid)
    Residual : final_output = decoder_output + input

    Returns
    -------
    autoencoder : full Model (input → reconstructed)
    encoder     : encoder sub-Model (input → bottleneck)
    decoder     : decoder sub-Model (bottleneck → reconstructed)
    """
    H, W, C = input_shape

    # ---- Encoder ----
    enc_input = layers.Input(shape=input_shape, name="encoder_input")

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="enc_conv1")(enc_input)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool1")(x)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="enc_pool2")(x)

    spatial_shape = (H // 4, W // 4, 16)
    x = layers.Flatten(name="enc_flatten")(x)
    bottleneck = layers.Dense(
        bottleneck_dim,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="bottleneck",
    )(x)
    bottleneck = layers.Dropout(0.2, name="bottleneck_dropout")(bottleneck)

    encoder = Model(enc_input, bottleneck, name="encoder")

    # ---- Decoder ----
    dec_input = layers.Input(shape=(bottleneck_dim,), name="decoder_input")

    flat_dim = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    y = layers.Dense(flat_dim, activation="relu", name="dec_dense")(dec_input)
    y = layers.Reshape(spatial_shape, name="dec_reshape")(y)

    y = layers.UpSampling2D((2, 2), name="dec_up1")(y)
    y = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="dec_conv1")(y)
    y = layers.BatchNormalization(name="dec_bn1")(y)

    y = layers.UpSampling2D((2, 2), name="dec_up2")(y)
    y = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="dec_conv2")(y)
    y = layers.BatchNormalization(name="dec_bn2")(y)

    dec_output = layers.Conv2D(C, (3, 3), padding="same", activation="sigmoid", name="dec_output")(y)

    decoder = Model(dec_input, dec_output, name="decoder")

    # ---- Full Autoencoder with Residual ----
    ae_input = layers.Input(shape=input_shape, name="ae_input")
    encoded = encoder(ae_input)
    decoded = decoder(encoded)

    # Residual connection: preserve fine-grained detail
    final_output = layers.Add(name="residual_add")([decoded, ae_input])
    final_output = layers.Activation("sigmoid", name="final_sigmoid")(final_output)

    autoencoder = Model(ae_input, final_output, name="sdae_autoencoder")
    return autoencoder, encoder, decoder


# ---------------------------------------------------------------------------
# 5. Combined Loss: MSE + SSIM
# ---------------------------------------------------------------------------

def combined_loss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.8) -> tf.Tensor:
    """
    Loss = alpha * MSE + (1 - alpha) * (1 - SSIM)

    Parameters
    ----------
    alpha : float  weight for MSE component (default 0.8)
    """
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return alpha * mse + (1.0 - alpha) * (1.0 - ssim)


# ---------------------------------------------------------------------------
# 6. Full Pipeline  (compress + decompress)
# ---------------------------------------------------------------------------

class HybridCompressor:
    """
    End-to-end hybrid compression pipeline.

    Usage
    -----
    >>> hc = HybridCompressor()
    >>> hc.build()
    >>> hc.train(X_train)
    >>> compressed = hc.compress(image)
    >>> reconstructed = hc.decompress(compressed)
    """

    def __init__(
        self,
        input_shape: tuple = (256, 256, 3),
        bottleneck_dim: int = 64,
        n_clusters: int = 4,
        wavelet: str = "haar",
        swt_level: int = 1,
        patch_size: int = 32,
    ):
        self.input_shape = input_shape
        self.bottleneck_dim = bottleneck_dim
        self.n_clusters = n_clusters
        self.wavelet = wavelet
        self.swt_level = swt_level
        self.patch_size = patch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.kmeans = None

    # ------------------------------------------------------------------
    def build(self):
        """Instantiate the SDAE models."""
        self.autoencoder, self.encoder, self.decoder = build_sdae(
            input_shape=self.input_shape,
            bottleneck_dim=self.bottleneck_dim,
        )
        self.autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=combined_loss,
            metrics=["mse"],
        )
        return self

    # ------------------------------------------------------------------
    def train(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.1,
        callbacks=None,
    ):
        """Train the SDAE on normalised images."""
        if self.autoencoder is None:
            self.build()

        # Also fit K-Means on training features
        print("[HybridCompressor] Extracting GLCM features for K-Means fitting …")
        all_features = []
        for img in X_train[:200]:   # use up to 200 images for speed
            feats, _ = extract_patch_glcm_features(img, self.patch_size)
            all_features.append(feats)
        all_features = np.vstack(all_features)
        _, self.kmeans = cluster_regions(all_features, n_clusters=self.n_clusters)
        print(f"[HybridCompressor] K-Means fitted with {self.n_clusters} clusters.")

        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks or [],
            verbose=1,
        )
        return history

    # ------------------------------------------------------------------
    def compress(self, image: np.ndarray) -> dict:
        """
        Compress a single image.

        Returns a dict with keys:
          bottleneck  – quantised latent vector
          swt_approx  – approximation coefficients (low-freq, not encoded)
          cluster_map – patch cluster labels
          orig_shape  – original image shape
        """
        assert self.encoder is not None, "Call build() / train() first."
        img = self._preprocess(image)

        # 1. GLCM + K-Means
        feats, coords = extract_patch_glcm_features(img, self.patch_size)
        if self.kmeans is not None:
            cluster_labels = self.kmeans.predict(feats)
        else:
            cluster_labels, _ = cluster_regions(feats, self.n_clusters)

        # 2. SWT per channel
        swt_coeffs = apply_swt(img, wavelet=self.wavelet, level=self.swt_level)

        # 3. SDAE encode
        img_batch = img[np.newaxis, ...]          # (1, H, W, C)
        bottleneck = self.encoder.predict(img_batch, verbose=0)[0]

        # 4. Quantise bottleneck (8-bit)
        q_bottleneck = np.round(bottleneck * 255) / 255

        return {
            "bottleneck": q_bottleneck,
            "swt_coeffs": swt_coeffs,
            "cluster_labels": cluster_labels,
            "patch_coords": coords,
            "orig_shape": image.shape,
        }

    # ------------------------------------------------------------------
    def decompress(self, compressed: dict) -> np.ndarray:
        """Reconstruct an image from compressed representation."""
        assert self.decoder is not None, "Call build() / train() first."

        bottleneck = compressed["bottleneck"][np.newaxis, ...]
        reconstructed = self.decoder.predict(bottleneck, verbose=0)[0]

        # Clamp to [0, 1]
        reconstructed = np.clip(reconstructed, 0.0, 1.0)
        return reconstructed

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save the full autoencoder weights."""
        self.autoencoder.save_weights(path)
        print(f"[HybridCompressor] Weights saved → {path}")

    def load(self, path: str):
        """Load saved weights."""
        if self.autoencoder is None:
            self.build()
        self.autoencoder.load_weights(path)
        print(f"[HybridCompressor] Weights loaded ← {path}")

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Resize to 256×256 and normalise to [0, 1]."""
        import cv2
        img = cv2.resize(image, (256, 256))
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img.astype(np.float32) / 255.0
