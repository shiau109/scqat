import numpy as np
import pytest

from scqat.math_tools.hankel import hankel_decompose, _select_n_modes


class TestHankelDecompose:
    """Tests for hankel_decompose using synthetic decaying-sinusoid signals."""

    def test_single_mode_extraction(self):
        """A single decaying cosine should yield one dominant mode with correct freq & decay."""
        freq, decay = 2.0, -0.3
        t = np.linspace(0, 10, 300)
        signal = np.exp(decay * t) * np.cos(2 * np.pi * freq * t)

        results = hankel_decompose(signal, t)

        modes = results["modes"]
        assert len(modes) >= 1
        dominant = modes[0]
        assert dominant["freq_hz"] == pytest.approx(freq, abs=0.1)
        assert dominant["decay_rate"] == pytest.approx(decay, abs=0.1)

    def test_two_mode_extraction(self):
        """Two decaying cosines should produce two dominant modes."""
        t = np.linspace(0, 10, 300)
        signal = (
            3.0 * np.exp(-0.5 * t) * np.cos(2 * np.pi * 1.5 * t)
            + 1.5 * np.exp(-0.2 * t) * np.cos(2 * np.pi * 0.8 * t)
        )

        results = hankel_decompose(signal, t)

        freqs = sorted([m["freq_hz"] for m in results["modes"][:2]])
        assert freqs[0] == pytest.approx(0.8, abs=0.15)
        assert freqs[1] == pytest.approx(1.5, abs=0.15)

    def test_reconstruction_quality(self):
        """Reconstruction error should be small for a clean signal."""
        t = np.linspace(0, 10, 300)
        signal = 2.0 * np.exp(-0.4 * t) * np.cos(2 * np.pi * 1.0 * t)

        results = hankel_decompose(signal, t)

        residual = np.abs(signal - results["reconstruction"])
        assert np.max(residual) < 0.5

    def test_hsvd_method(self):
        """HSVD reconstruction method should also return valid modes."""
        t = np.linspace(0, 10, 300)
        signal = np.exp(-0.3 * t) * np.cos(2 * np.pi * 2.0 * t)

        results = hankel_decompose(signal, t, recon_method="hsvd")

        assert len(results["modes"]) >= 1
        assert results["modes"][0]["freq_hz"] == pytest.approx(2.0, abs=0.2)

    # ------------------------------------------------------------------
    # Mode-selection strategies
    # ------------------------------------------------------------------
    def test_gap_mode_selection(self):
        """Gap method should pick up the dominant mode for a clean signal."""
        t = np.linspace(0, 10, 300)
        signal = np.exp(-0.3 * t) * np.cos(2 * np.pi * 2.0 * t)

        results = hankel_decompose(signal, t, mode_method="gap")

        assert results["n_modes"] >= 1
        assert results["modes"][0]["freq_hz"] == pytest.approx(2.0, abs=0.2)

    def test_aic_mode_selection(self):
        """AIC method should extract a valid dominant mode."""
        t = np.linspace(0, 10, 300)
        signal = np.exp(-0.3 * t) * np.cos(2 * np.pi * 2.0 * t)

        results = hankel_decompose(signal, t, mode_method="aic")

        assert results["n_modes"] >= 1
        assert results["modes"][0]["freq_hz"] == pytest.approx(2.0, abs=0.2)

    def test_mdl_mode_selection(self):
        """MDL method should extract a valid dominant mode."""
        t = np.linspace(0, 10, 300)
        signal = np.exp(-0.3 * t) * np.cos(2 * np.pi * 2.0 * t)

        results = hankel_decompose(signal, t, mode_method="mdl")

        assert results["n_modes"] >= 1
        assert results["modes"][0]["freq_hz"] == pytest.approx(2.0, abs=0.2)

    def test_fixed_mode_selection(self):
        """Fixed method should use the exact number of modes requested."""
        t = np.linspace(0, 10, 300)
        signal = np.exp(-0.3 * t) * np.cos(2 * np.pi * 2.0 * t)

        results = hankel_decompose(signal, t, mode_method="fixed", n_modes=3)

        assert results["n_modes"] == 3

    def test_fixed_mode_missing_n_modes_raises(self):
        """Fixed mode selection without n_modes kwarg should raise ValueError."""
        s = np.array([10.0, 5.0, 1.0, 0.1])
        with pytest.raises(ValueError, match="n_modes"):
            _select_n_modes(s, method="fixed")

    def test_unknown_mode_method_raises(self):
        """Unknown mode selection method should raise ValueError."""
        s = np.array([10.0, 5.0, 1.0])
        with pytest.raises(ValueError, match="Unknown"):
            _select_n_modes(s, method="bogus")
