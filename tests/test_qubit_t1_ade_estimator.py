"""QubitT1AdeEstimator: recovery, clip mask, bootstrap, artifacts, replot."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qubit_t1_ade import QubitT1AdeEstimator
from scqat.tools.ade_decay import ade_gamma, ade_sigma_gamma

T0 = 1e-6
DT = 30e-6


def _ade_ds(t1=40e-6, n_blocks=20, n_avg=400, dt=DT, with_shots=True,
            with_time=True, seed=9):
    """Plant a T1; draw the per-shot outcomes; compute the FPGA streams FROM
    those shots (as the hardware does), so the offline recompute matches."""
    rng = np.random.default_rng(seed)
    gamma_true = 1.0 / t1
    p = 0.93 * np.exp(-gamma_true * np.array([T0, T0 + dt, T0 + 3 * dt])) + 0.05
    shots = rng.binomial(1, p[:, None], size=(n_blocks, 3, n_avg)).astype(np.int8)
    pops = shots.mean(axis=2)
    dt_arr = np.full(n_blocks, dt)
    gamma, _ = ade_gamma(pops[:, 0], pops[:, 1], pops[:, 2], dt_arr)
    sigma = ade_sigma_gamma(pops[:, 0], pops[:, 1], pops[:, 2], dt_arr, n_avg)

    data_vars = {
        "estimated_gamma": ("block_idx", gamma),
        "sigma_gamma": ("block_idx", sigma),
        "dt_s": ("block_idx", dt_arr),
    }
    if with_shots:
        data_vars["state"] = (("block_idx", "delay_idx", "shot_idx"), shots)
    coords = {"block_idx": np.arange(n_blocks)}
    if with_time:
        data_vars["block_time_s"] = ("block_idx", np.arange(n_blocks) * 0.05)
    return xr.Dataset(data_vars, coords=coords)


class TestQubitT1AdeEstimator:
    def test_recovery(self):
        t1 = 40e-6
        results, _ = QubitT1AdeEstimator().analyze(_ade_ds(t1=t1), skip_figures=True)
        assert results["success"]
        assert results["t1_median_s"] == pytest.approx(t1, rel=0.1)
        assert results["n_clipped"] == 0
        # FPGA streams were computed from the same shots -> exact agreement
        assert results["n_fpga_mismatch"] == 0
        assert np.isfinite(results["t1_sigma_median_s"])

    def test_bootstrap_sigma(self):
        results, _ = QubitT1AdeEstimator().analyze(
            _ade_ds(n_blocks=6), skip_figures=True,
            n_bootstrap=200, bootstrap_seed=2,
        )
        assert np.isfinite(results["t1_boot_sigma_median_s"])
        # both sigmas estimate the same shot noise — loose factor-2 agreement
        assert results["t1_boot_sigma_median_s"] == pytest.approx(
            results["t1_sigma_median_s"], rel=1.0
        )

    def test_clipped_block_is_masked(self):
        """A block outside the validity domain (no decay resolved) is flagged
        and excluded from the median instead of poisoning it."""
        ds = _ade_ds(n_blocks=8)
        state = ds["state"].values.copy()
        state[3] = 1  # all delays fully excited -> c undefined
        ds["state"] = (("block_idx", "delay_idx", "shot_idx"), state)
        # the FPGA stream for that block is a clip-floored plausible number
        gamma = ds["estimated_gamma"].values.copy()
        gamma[3] = 1.0 / 1e-6
        ds["estimated_gamma"] = ("block_idx", gamma)
        results, _ = QubitT1AdeEstimator().analyze(ds, skip_figures=True)
        assert results["n_clipped"] == 1
        assert bool(results["clipped"][3])
        assert results["t1_median_s"] == pytest.approx(40e-6, rel=0.1)

    def test_works_without_shots(self):
        results, _ = QubitT1AdeEstimator().analyze(
            _ade_ds(with_shots=False), skip_figures=True
        )
        assert results["success"]
        assert results["has_shots"] is False
        assert np.isnan(results["t1_boot_sigma_median_s"])

    def test_unknown_kwarg_raises(self):
        with pytest.raises(ValueError, match="Unknown qubit_t1_ade kwarg"):
            QubitT1AdeEstimator().analyze(_ade_ds(), skip_figures=True, bootstrap=5)

    def test_metadata_drops_arrays(self):
        est = QubitT1AdeEstimator()
        results = est.extract_parameters(_ade_ds())
        metadata = est.extract_metadata(results)
        assert "t1_s" not in metadata
        assert "clipped" not in metadata
        assert {"t1_median_s", "n_valid", "success"} <= set(metadata)

    def test_imports_match(self):
        from scqat.estimators import QubitT1AdeEstimator as FromAggregate

        assert FromAggregate is QubitT1AdeEstimator
        assert QubitT1AdeEstimator.estimator_name == "qubit_t1_ade"

    def test_artifacts_and_figure_names(self, tmp_path):
        """Single-figure idiom: the trace file is qubit_t1_ade.png."""
        QubitT1AdeEstimator().analyze(_ade_ds(), output_dir=str(tmp_path))
        names = {p.name for p in tmp_path.iterdir()}
        assert "qubit_t1_ade_metadata.json" in names
        assert "qubit_t1_ade_plotdata.nc" in names
        assert "qubit_t1_ade.png" in names
        assert "qubit_t1_ade_dt_trace.png" in names
        plt.close("all")

    def test_check_data_rejects_missing(self):
        with pytest.raises(ValueError, match="estimated_gamma"):
            QubitT1AdeEstimator().analyze(
                xr.Dataset({"other": ("block_idx", [1.0])},
                           coords={"block_idx": [0]})
            )

    def test_figures_render_on_a_failed_fit(self):
        """Every block invalid -> success False, but both figures still render
        (SCQO's artifact fallback drops ALL figures on one plotter crash)."""
        n = 5
        ds = xr.Dataset(
            {
                "estimated_gamma": ("block_idx", np.full(n, np.nan)),
                "sigma_gamma": ("block_idx", np.full(n, np.nan)),
                "dt_s": ("block_idx", np.full(n, DT)),
            },
            coords={"block_idx": np.arange(n)},
        )
        est = QubitT1AdeEstimator()
        res = est.extract_parameters(ds)
        assert res["success"] is False
        pd = est.build_plot_data(ds, res)
        figs = est.generate_figures(None, None, plot_data=pd)
        assert {"qubit_t1_ade", "dt_trace"} <= set(figs)
        plt.close("all")

    def test_replot_roundtrip(self, tmp_path):
        est = QubitT1AdeEstimator()
        _, figs = est.analyze(_ade_ds(), output_dir=str(tmp_path))
        loaded = est.load_plot_data(str(tmp_path))
        refigs = est.generate_figures(None, None, plot_data=loaded)
        assert set(refigs) == set(figs)
        plt.close("all")
