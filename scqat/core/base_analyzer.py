import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import xarray as xr


class BaseAnalyzer(ABC):
    """
    Abstract base class for scqat experimental protocols.
    Enforces a strict separation of Data Checking, Math, Visualization, and I/O.

    Subclasses must define ``protocol_name`` (str) to control default
    output filenames when ``output_dir`` is used.
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
        """Step 1: The heavy calculation. Must return a results dictionary."""
        pass

    @abstractmethod
    def generate_figures(self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs) -> Dict[str, plt.Figure]:
        """Step 2: The visualization. Must return a dictionary of figures."""
        pass

    def save_metadata(self, results: Dict[str, Any], output_dir: str) -> None:
        """Saves the results dictionary as ``<output_dir>/<protocol_name>_results.pkl``."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"{self.protocol_name}_results.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

    def load_metadata(self, output_dir: str) -> Dict[str, Any]:
        """Loads the results dictionary from ``<output_dir>/<protocol_name>_results.pkl``."""
        filepath = os.path.join(output_dir, f"{self.protocol_name}_results.pkl")
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def save_figures(self, figs: Dict[str, plt.Figure], output_dir: str) -> None:
        """Saves figures as ``<output_dir>/<protocol_name>_<fig_name>.png``."""
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figs.items():
            filepath = os.path.join(output_dir, f"{self.protocol_name}_{name}.png")
            fig.savefig(filepath, bbox_inches='tight')

    def analyze(
        self, 
        dataset: xr.Dataset, 
        output_dir: str = None, 
        skip_figures: bool = False,
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
        """
        The Orchestrator. 
        Calls Data Checking -> Math -> Metadata I/O -> Plotting -> Figure I/O.

        Args:
            dataset: The input xarray Dataset.
            output_dir: Directory path for saving results and figures.
                        If None, nothing is saved.
            skip_figures: If True, skip figure generation and return empty dict.
        """
        # 1. Input checking
        self._check_data(dataset)
        
        # 2. Heavy physics calculation
        results = self.extract_parameters(dataset, **kwargs)
        
        # 3. Save metadata if requested
        if output_dir:
            self.save_metadata(results, output_dir)

        if skip_figures:
            return results, {}
            
        # 4. Generate figures
        figs = self.generate_figures(dataset, results, **kwargs)
        
        # 5. Save figures if requested
        if output_dir:
            self.save_figures(figs, output_dir)
            
        return results, figs