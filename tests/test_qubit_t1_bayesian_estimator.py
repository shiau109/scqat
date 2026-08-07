"""QubitT1BayesianEstimator: k reconstruction, CI, validation fit, artifacts."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qubit_t1_bayesian import QubitT1BayesianEstimator

T1_TRUE = 30e-6
K_FINAL = 25.0


def _bayes_ds(t1=T1_TRUE, n_blocks=32, n_probes=40, k_final=K_FINAL,
              with_evol=True, with_validation=True, with_time=True,
              t1_trace_scale=1.0, seed=13):
    rng = np.random.default_rng(seed)
    t1_trace = t1_trace_scale * t1 * (1.0 + 0.03 * rng.standard_normal(n_blocks))
    data_vars = {
        "estimated_t1_s": ("block_idx", t1_trace),
        "u_final": ("block_idx", np.full(n_blocks, 1.0 / k_final)),
    }
    coords = {"block_idx": np.arange(n_blocks)}
    if with_time:
        data_vars["block_time_s"] = ("block_idx", np.arange(n_blocks) * 0.02)
    if with_evol:
        k_ev = np.linspace(1.0, k_final, n_probes)
        t1_ev = t1 * (1.0 + 0.5 * np.exp(-np.arange(n_probes) / 5.0))
        data_vars["u_evol"] = ("probe_idx", 1.0 / k_ev)
        data_vars["t1_evol_s"] = ("probe_idx", t1_ev)
    if with_validation:
        lin_wait = np.linspace(1e-6, 150e-6, n_probes)
        p = np.exp(-lin_wait / t1)
        state_lin = rng.binomial(1, p, size=(n_blocks, n_probes)).astype(np.int8)
        data_vars["state_lin"] = (("block_idx", "probe_idx"), state_lin)
        data_vars["lin_wait_s"] = ("probe_idx", lin_wait)
    return xr.Dataset(data_vars, coords=coords)


class TestQubitT1BayesianEstimator:
    def test_recovery_and_k_reconstruction(self):
        results, _ = QubitT1BayesianEstimator().analyze(_bayes_ds(), skip_figures=True)
        assert results["success"]
        assert results["t1_median_s"] == pytest.approx(T1_TRUE, rel=0.1)
        assert results["k_final_median"] == pytest.approx(K_FINAL, rel=1e-6)

    def test_credible_interval_straddles_the_estimate(self):
        results, _ = QubitT1BayesianEstimator().analyze(_bayes_ds(), skip_figures=True)
        t1 = results["t1_s_trace"]
        lo = results["t1_ci_low_s"]
        hi = results["t1_ci_high_s"]
        ok = np.isfinite(t1)
        assert np.all(lo[ok] < t1[ok]) and np.all(t1[ok] < hi[ok])
        # the band tightens as ~1/sqrt(k): at k=25 the half-width is well
        # under the estimate itself
        assert np.all((hi[ok] - lo[ok]) < t1[ok])

    def test_validation_fit_and_agreement(self):
        results, _ = QubitT1BayesianEstimator().analyze(_bayes_ds(), skip_figures=True)
        assert results["has_validation"]
        assert results["t1_lin_s"] == pytest.approx(T1_TRUE, rel=0.15)
        assert results["t1_lin_ratio"] == pytest.approx(1.0, rel=0.2)
        assert results["validation_disagrees"] is False

    def test_disagreement_is_flagged(self):
        """An adaptive trace pinned 3.5x above the classical decay — the
        reference node's symptom of a wrong prior — must raise the flag."""
        results, _ = QubitT1BayesianEstimator().analyze(
            _bayes_ds(t1_trace_scale=3.5), skip_figures=True
        )
        assert results["validation_disagrees"] is True

    def test_psd_and_allan_from_lab_time(self):
        results, _ = QubitT1BayesianEstimator().analyze(_bayes_ds(), skip_figures=True)
        assert results["psd_freq_hz"].size > 0
        assert results["allan_tau_s"].size > 0
        assert np.isfinite(results["psd_dt_s"])

    def test_no_lab_time_no_psd(self):
        results, _ = QubitT1BayesianEstimator().analyze(
            _bayes_ds(with_time=False), skip_figures=True
        )
        assert results["has_lab_time"] is False
        assert results["psd_freq_hz"].size == 0

    def test_ci_kwarg_validated(self):
        with pytest.raises(ValueError, match="ci"):
            QubitT1BayesianEstimator().analyze(_bayes_ds(), skip_figures=True, ci=1.5)

    def test_unknown_kwarg_raises(self):
        with pytest.raises(ValueError, match="Unknown timeseries-PSD knob"):
            QubitT1BayesianEstimator().analyze(_bayes_ds(), skip_figures=True, foo=1)

    def test_metadata_drops_arrays(self):
        est = QubitT1BayesianEstimator()
        results = est.extract_parameters(_bayes_ds())
        metadata = est.extract_metadata(results)
        assert "t1_s_trace" not in metadata
        assert "posterior_pdf" not in metadata
        assert {"t1_median_s", "k_final_median", "success"} <= set(metadata)

    def test_imports_match(self):
        from scqat.estimators import QubitT1BayesianEstimator as FromAggregate

        assert FromAggregate is QubitT1BayesianEstimator
        assert QubitT1BayesianEstimator.estimator_name == "qubit_t1_bayesian"

    def test_artifacts_and_figure_names(self, tmp_path):
        """Single-figure idiom: the trace file is qubit_t1_bayesian.png."""
        _, figs = QubitT1BayesianEstimator().analyze(
            _bayes_ds(), output_dir=str(tmp_path)
        )
        assert set(figs) == {"qubit_t1_bayesian", "posterior_evolution",
                             "psd", "allan", "validation"}
        names = {p.name for p in tmp_path.iterdir()}
        assert "qubit_t1_bayesian_metadata.json" in names
        assert "qubit_t1_bayesian_plotdata.nc" in names
        assert "qubit_t1_bayesian.png" in names
        assert "qubit_t1_bayesian_validation.png" in names
        plt.close("all")

    def test_check_data_rejects_missing(self):
        with pytest.raises(ValueError, match="estimated_t1_s"):
            QubitT1BayesianEstimator().analyze(
                xr.Dataset({"other": ("block_idx", [1.0])},
                           coords={"block_idx": [0]})
            )

    def test_figures_render_on_a_failed_fit(self):
        """All-NaN streams -> success False, but the trace figure still
        renders (SCQO's artifact fallback drops ALL figures on one crash)."""
        n = 6
        ds = xr.Dataset(
            {
                "estimated_t1_s": ("block_idx", np.full(n, np.nan)),
                "u_final": ("block_idx", np.full(n, np.nan)),
            },
            coords={"block_idx": np.arange(n)},
        )
        est = QubitT1BayesianEstimator()
        res = est.extract_parameters(ds)
        assert res["success"] is False
        pd = est.build_plot_data(ds, res)
        figs = est.generate_figures(None, None, plot_data=pd)
        assert "qubit_t1_bayesian" in figs
        plt.close("all")

    def test_replot_roundtrip(self, tmp_path):
        est = QubitT1BayesianEstimator()
        _, figs = est.analyze(_bayes_ds(), output_dir=str(tmp_path))
        loaded = est.load_plot_data(str(tmp_path))
        refigs = est.generate_figures(None, None, plot_data=loaded)
        assert set(refigs) == set(figs)
        plt.close("all")
