"""Tests for the figure-isolation helper (raw data always plottable)."""

import warnings

import matplotlib.pyplot as plt
import pytest

from scqat.core.figures import render_figures


def _fig():
    return plt.figure()


def test_all_builders_succeed_returns_all():
    figs = render_figures({"a": _fig, "b": _fig})
    assert set(figs) == {"a", "b"}
    for f in figs.values():
        assert isinstance(f, plt.Figure)
        plt.close(f)


def test_a_failing_builder_is_skipped_and_the_others_survive():
    def boom():
        raise ValueError("Data cannot be log-scaled because all values are <= 0.")

    with pytest.warns(UserWarning, match="figure 'fit' failed to render"):
        figs = render_figures({"fit": boom, "raw": _fig}, label="spectroscopy_cryoscope")

    # the raw figure survives even though the fit figure raised
    assert set(figs) == {"raw"}
    assert isinstance(figs["raw"], plt.Figure)
    plt.close(figs["raw"])


def test_label_appears_in_the_warning():
    def boom():
        raise RuntimeError("nope")

    with pytest.warns(UserWarning, match="my_estimator: figure 'x' failed"):
        render_figures({"x": boom}, label="my_estimator")


def test_no_label_defaults_to_estimator_in_the_message():
    def boom():
        raise RuntimeError("nope")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        render_figures({"x": boom})
    assert any("estimator: figure 'x' failed" in str(w.message) for w in caught)
