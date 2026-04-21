"""
tests/test_model.py
--------------------
Unit tests for the Hybrid Image Compression model components.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image():
    """256×256 RGB image, float32 in [0, 1]."""
    rng = np.random.default_rng(42)
    return rng.random((256, 256, 3), dtype=np.float32)


@pytest.fixture
def dummy_batch():
    rng = np.random.default_rng(0)
    return rng.random((4, 256, 256, 3), dtype=np.float32)


# ---------------------------------------------------------------------------
# GLCM tests
# ---------------------------------------------------------------------------

class TestGLCM:
    def test_feature_vector_length(self, dummy_image):
        from src.model import compute_glcm_features
        feats = compute_glcm_features(dummy_image)
        # 5 features × 3 channels = 15
        assert feats.shape == (15,), f"Expected (15,), got {feats.shape}"

    def test_features_finite(self, dummy_image):
        from src.model import compute_glcm_features
        feats = compute_glcm_features(dummy_image)
        assert np.all(np.isfinite(feats)), "GLCM features contain NaN or Inf"

    def test_patch_extraction_shape(self, dummy_image):
        from src.model import extract_patch_glcm_features
        feats, coords = extract_patch_glcm_features(dummy_image, patch_size=32)
        n_patches = (256 // 32) ** 2  # 64 patches
        assert feats.shape[0] == n_patches
        assert len(coords) == n_patches

    def test_grayscale_input(self):
        from src.model import compute_glcm_features
        gray = np.random.rand(64, 64).astype(np.float32)
        feats = compute_glcm_features(gray)
        # 5 features × 1 channel = 5
        assert feats.shape == (5,)


# ---------------------------------------------------------------------------
# K-Means clustering tests
# ---------------------------------------------------------------------------

class TestKMeans:
    def test_labels_range(self, dummy_image):
        from src.model import extract_patch_glcm_features, cluster_regions
        feats, _ = extract_patch_glcm_features(dummy_image)
        labels, _ = cluster_regions(feats, n_clusters=4)
        assert set(labels).issubset({0, 1, 2, 3})

    def test_labels_count(self, dummy_image):
        from src.model import extract_patch_glcm_features, cluster_regions
        feats, _ = extract_patch_glcm_features(dummy_image)
        labels, _ = cluster_regions(feats, n_clusters=4)
        assert len(labels) == len(feats)

    def test_kmeans_predict(self, dummy_image):
        from src.model import extract_patch_glcm_features, cluster_regions
        feats, _ = extract_patch_glcm_features(dummy_image)
        labels, kmeans = cluster_regions(feats, n_clusters=3)
        new_pred = kmeans.predict(feats[:5])
        assert new_pred.shape == (5,)


# ---------------------------------------------------------------------------
# SWT tests
# ---------------------------------------------------------------------------

class TestSWT:
    def test_swt_output_keys(self, dummy_image):
        from src.model import apply_swt
        coeffs = apply_swt(dummy_image, wavelet="haar", level=1)
        # Multi-channel: list of length 3 (one per channel)
        assert len(coeffs) == 3

    def test_swt_iswt_roundtrip(self):
        """SWT → ISWT should reconstruct the original array."""
        from src.model import apply_swt, apply_iswt
        import pywt
        gray = np.random.rand(64, 64).astype(np.float64)
        coeffs = pywt.swt2(gray, wavelet="haar", level=1)
        recon = pywt.iswt2(coeffs, wavelet="haar")
        assert np.allclose(gray, recon, atol=1e-5), "SWT round-trip failed"

    def test_swt_preserves_shape(self, dummy_image):
        """SWT approximation sub-band should match input spatial dims."""
        from src.model import apply_swt
        coeffs = apply_swt(dummy_image[:, :, 0], wavelet="haar", level=1)
        cA, _ = coeffs[0]
        assert cA.shape == dummy_image.shape[:2]


# ---------------------------------------------------------------------------
# SDAE tests
# ---------------------------------------------------------------------------

class TestSDAE:
    def test_build_returns_three_models(self):
        from src.model import build_sdae
        ae, enc, dec = build_sdae(input_shape=(256, 256, 3), bottleneck_dim=32)
        assert ae is not None
        assert enc is not None
        assert dec is not None

    def test_encoder_output_shape(self, dummy_image):
        from src.model import build_sdae
        _, enc, _ = build_sdae(input_shape=(256, 256, 3), bottleneck_dim=32)
        out = enc.predict(dummy_image[np.newaxis], verbose=0)
        assert out.shape == (1, 32)

    def test_decoder_output_shape(self):
        from src.model import build_sdae
        _, _, dec = build_sdae(input_shape=(256, 256, 3), bottleneck_dim=32)
        z = np.random.rand(1, 32).astype(np.float32)
        out = dec.predict(z, verbose=0)
        assert out.shape == (1, 256, 256, 3)

    def test_autoencoder_output_range(self, dummy_image):
        from src.model import build_sdae
        ae, _, _ = build_sdae(input_shape=(256, 256, 3), bottleneck_dim=32)
        out = ae.predict(dummy_image[np.newaxis], verbose=0)[0]
        assert out.min() >= -0.01, "Output below 0"
        assert out.max() <= 1.01, "Output above 1"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_psnr_identical(self, dummy_image):
        from src.metrics import psnr
        score = psnr(dummy_image, dummy_image)
        assert score > 100, f"PSNR for identical images should be very high, got {score}"

    def test_ssim_identical(self, dummy_image):
        from src.metrics import ssim
        score = ssim(dummy_image, dummy_image)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_prd_identical(self, dummy_image):
        from src.metrics import prd
        score = prd(dummy_image, dummy_image)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_bpp_calculation(self):
        from src.metrics import bpp
        score = bpp(compressed_bytes=10000, image_shape=(256, 256))
        expected = (10000 * 8) / (256 * 256)
        assert score == pytest.approx(expected, rel=1e-5)

    def test_evaluate_all_keys(self, dummy_image):
        from src.metrics import evaluate_all
        noisy = np.clip(dummy_image + 0.1 * np.random.rand(*dummy_image.shape), 0, 1).astype(np.float32)
        m = evaluate_all(dummy_image, noisy, compressed_bytes=5000)
        for key in ["psnr", "ssim", "ms_ssim", "mse", "prd", "bpp"]:
            assert key in m, f"Missing key: {key}"

    def test_mse_positive(self, dummy_image):
        from src.metrics import mse
        noisy = np.clip(dummy_image + 0.05, 0, 1).astype(np.float32)
        assert mse(dummy_image, noisy) > 0


# ---------------------------------------------------------------------------
# Combined loss tests
# ---------------------------------------------------------------------------

class TestLoss:
    def test_loss_zero_for_identical(self):
        import tensorflow as tf
        from src.model import combined_loss
        x = tf.constant(np.random.rand(1, 256, 256, 3).astype(np.float32))
        loss = combined_loss(x, x).numpy()
        assert loss < 1e-5, f"Loss for identical tensors should be ~0, got {loss}"

    def test_loss_positive(self):
        import tensorflow as tf
        from src.model import combined_loss
        x = tf.constant(np.random.rand(1, 256, 256, 3).astype(np.float32))
        y = tf.constant(np.random.rand(1, 256, 256, 3).astype(np.float32))
        loss = combined_loss(x, y).numpy()
        assert loss > 0
