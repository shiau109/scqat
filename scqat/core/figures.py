"""Figure isolation — how every estimator keeps its raw data plottable.

An estimator draws several figures from one ``plot_data`` (a raw-data view, a
fit view, a diagnostic). They must be INDEPENDENT: a fit that failed, or a
plotter that raises on a degenerate (all-NaN) fit, must never take the raw-data
figure down with it. :func:`render_figures` enforces that — each figure is built
in its own ``try``/``except``, and a builder that raises is skipped with a
warning while the others still render.

Estimators use it in :meth:`BaseEstimator.generate_figures` by returning
``render_figures({name: lambda: plot_x(plot_data), ...}, label=self.estimator_name)``
— note the values are THUNKS (zero-arg callables), so a builder that raises does
so INSIDE the isolation, not while the dict is being assembled.

The rule this implements ("raw data must always be plottable") is stated in
``scqat/CLAUDE.md`` under the Estimator Output Contract.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict

import matplotlib.pyplot as plt


def render_figures(
    builders: Dict[str, Callable[[], plt.Figure]], *, label: str = ""
) -> Dict[str, plt.Figure]:
    """Build each figure in isolation; return the ones that succeed.

    Parameters
    ----------
    builders : dict[str, callable]
        ``{figure_name: thunk}`` where each thunk takes no arguments and returns
        a Matplotlib ``Figure``. Pass thunks (``lambda: plot_x(plot_data)``), not
        pre-built figures, so a builder that raises is caught here rather than
        while the caller assembles the dict.
    label : str
        Estimator name, used only in the warning message.

    Returns
    -------
    dict[str, plt.Figure]
        Only the figures that built without raising. A builder that raises is
        omitted and a :class:`UserWarning` is emitted naming it — the run keeps
        every figure that could be drawn (crucially, the raw-data one) instead of
        losing them all to a single broken plotter.
    """
    figures: Dict[str, plt.Figure] = {}
    for name, build in builders.items():
        try:
            figures[name] = build()
        except Exception as err:  # noqa: BLE001 - a plotter must never sink the run
            warnings.warn(
                f"{label or 'estimator'}: figure '{name}' failed to render "
                f"({type(err).__name__}: {err}); skipping it",
                stacklevel=2,
            )
    return figures
