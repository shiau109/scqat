import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _json_safe(obj: Any) -> Any:
    """
    Recursively convert ``obj`` into something ``json.dump`` can serialize.

    numpy scalars/arrays become Python scalars/lists, complex numbers become
    ``{"real": ..., "imag": ...}``, and objects that cannot be represented as
    plain metadata (e.g. ``xarray`` containers, lmfit results) are dropped with
    a short ``"<skipped: type>"`` marker so the metadata file never fails to
    write.  Bulky arrays belong in the plot-data Dataset, not here.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # NaN/Inf are not valid JSON — map to null for cross-language portability.
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (complex, np.complexfloating)):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (xr.Dataset, xr.DataArray)):
        return f"<skipped: {type(obj).__name__} — belongs in plot_data>"
    return f"<skipped: {type(obj).__name__}>"


class BaseAnalyzer(ABC):
    """
    Abstract base class for scqat experimental/simulation protocols.

    Enforces a strict separation of Data Checking, Math, Plot-data extraction,
    Visualization, and I/O.  An analyzer produces **one mandatory artifact and
    two optional ones**:

    * **metadata** (mandatory) — the key physical parameters. Computed by
      :meth:`extract_parameters` and projected by :meth:`extract_metadata`,
      saved as ``<protocol_name>_metadata.json``.
    * **plot data** (optional) — the minimal arrays needed to redraw every
      figure with no recalculation, returned by :meth:`build_plot_data` and
      saved as ``<protocol_name>_plotdata.nc``.
    * **figures** (optional) — produced by :meth:`generate_figures`, which draws
      **only** from the plot data. Providing figures therefore implies providing
      plot data.

    The only method a subclass MUST implement is :meth:`extract_parameters`;
    everything else has a safe default. Subclasses must also define
    ``protocol_name`` (str) to control default output filenames.
    """

    protocol_name: str = "protocol"

    def _check_data(self, dataset: xr.Dataset) -> None:
        """
        Optional data validation step.
        Override this in your subclass to check for required coordinates/variables.
        """
        pass

    @abstractmethod
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Step 1 (mandatory): The heavy calculation. Returns the analysis
        ``results`` — the key parameters plus any rich intermediates that
        :meth:`build_plot_data` needs. For a simple protocol this dict *is* the
        metadata.
        """
        pass

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Project ``results`` down to the key parameters that get persisted as
        JSON metadata.

        Default is the identity, so a simple protocol's ``results`` is saved
        verbatim. Override when ``results`` carries bulky intermediates (large
        arrays, ``xarray`` containers, fit objects) that should not land in the
        metadata file — return just the key scalars/arrays.
        """
        return results

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Step 2 (optional): Assemble the minimal arrays needed to redraw every
        figure without any recalculation, as a single ``xarray.Dataset``.

        Default returns ``None`` (no plot-data artifact). Override to provide a
        self-sufficient Dataset; :meth:`generate_figures` should then draw using
        only this Dataset.
        """
        return None

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """
        Step 3 (optional): The visualization. Returns a dict of figures.

        Migrated protocols draw using **only** ``plot_data`` so the figures stay
        reconstructable by an external consumer; ``dataset`` and ``results`` are
        still passed for protocols not yet migrated to the plot-data contract and
        must be ignored by new code.

        Default returns ``{}`` (no figures). Override only alongside
        :meth:`build_plot_data` — figures require plot data.
        """
        return {}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def save_metadata(self, metadata: Dict[str, Any], output_dir: str) -> None:
        """
        Save the key parameters as ``<output_dir>/<protocol_name>_metadata.json``.

        The file is stamped with ``protocol_name`` so a loaded metadata file
        self-identifies which analyzer produced it, even as the parameter set
        evolves over time.
        """
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.protocol_name}_metadata.json")
        payload = {"protocol_name": self.protocol_name, **_json_safe(metadata)}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_metadata(self, output_dir: str) -> Dict[str, Any]:
        """Load the key parameters from ``<output_dir>/<protocol_name>_metadata.json``."""
        filepath = os.path.join(output_dir, f"{self.protocol_name}_metadata.json")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_plot_data(self, plot_data: Optional[xr.Dataset], output_dir: str) -> None:
        """Save the plot-reconstruction Dataset as ``<output_dir>/<protocol_name>_plotdata.nc``."""
        if plot_data is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.protocol_name}_plotdata.nc")
        plot_data.to_netcdf(filepath)

    def load_plot_data(self, output_dir: str) -> xr.Dataset:
        """Load the plot-reconstruction Dataset from ``<output_dir>/<protocol_name>_plotdata.nc``."""
        filepath = os.path.join(output_dir, f"{self.protocol_name}_plotdata.nc")
        return xr.load_dataset(filepath)

    def save_figures(self, figs: Dict[str, plt.Figure], output_dir: str) -> None:
        """Saves figures as ``<output_dir>/<protocol_name>_<fig_name>.png``."""
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figs.items():
            filepath = os.path.join(output_dir, f"{self.protocol_name}_{name}.png")
            fig.savefig(filepath, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def analyze(
        self,
        dataset: xr.Dataset,
        output_dir: str = None,
        skip_figures: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
        """
        The Orchestrator.

        Calls Data Checking -> Math (results) -> Metadata projection ->
        Plot-data build -> Metadata/Plot-data I/O -> Plotting -> Figure I/O.

        The metadata artifact is mandatory; plot data and figures are produced
        only when the analyzer overrides :meth:`build_plot_data` /
        :meth:`generate_figures` (otherwise nothing is saved/drawn for them).

        Args:
            dataset: The input xarray Dataset.
            output_dir: Directory path for saving metadata, plot data, and
                figures. If None, nothing is saved.
            skip_figures: If True, skip figure generation and return empty dict.

        Returns:
            ``(results, figures)``. ``results`` is the full in-memory analysis
            output; the persisted metadata is ``extract_metadata(results)``. The
            plot-data Dataset is saved (when ``output_dir`` is given) and passed
            to ``generate_figures``; retrieve it via :meth:`build_plot_data` or
            :meth:`load_plot_data` if needed.
        """
        # 1. Input checking
        self._check_data(dataset)

        # 2. Heavy physics calculation -> full analysis results
        results = self.extract_parameters(dataset, **kwargs)

        # 3. Project the key parameters to persist (mandatory artifact)
        metadata = self.extract_metadata(results)

        # 4. Minimal arrays needed to redraw the figures (optional plot data)
        plot_data = self.build_plot_data(dataset, results, **kwargs)

        # 5. Save metadata + plot data if requested (plot data skipped when None)
        if output_dir:
            self.save_metadata(metadata, output_dir)
            self.save_plot_data(plot_data, output_dir)

        if skip_figures:
            return results, {}

        # 6. Generate figures (optional). Migrated protocols use only plot_data;
        #    dataset and results remain available for not-yet-migrated protocols.
        figs = self.generate_figures(dataset, results, plot_data=plot_data, **kwargs)

        # 7. Save figures if requested
        if output_dir:
            self.save_figures(figs, output_dir)

        return results, figs
