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
    """

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

    def save_metadata(self, results: Dict[str, Any], filepath: str) -> None:
        """Saves the results dictionary safely using pickle."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)

    def load_metadata(self, filepath: str) -> Dict[str, Any]:
        """Loads the exact results dictionary back into memory."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def save_figures(self, figs: Dict[str, plt.Figure], base_path: str) -> None:
        """Saves the dictionary of Matplotlib figures to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(f"{base_path}_{name}.png", bbox_inches='tight')

    def analyze(
        self, 
        dataset: xr.Dataset, 
        metadata_save_path: str = None, 
        figure_save_base_path: str = None, 
        **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
        """
        The Orchestrator. 
        Calls Data Checking -> Math -> Metadata I/O -> Plotting -> Figure I/O.
        """
        # 1. Input checking
        self._check_data(dataset)
        
        # 2. Heavy physics calculation
        results = self.extract_parameters(dataset, **kwargs)
        
        # 3. Save metadata if requested
        if metadata_save_path:
            self.save_metadata(results, metadata_save_path)
            
        # 4. Generate figures
        figs = self.generate_figures(dataset, results, **kwargs)
        
        # 5. Save figures if requested
        if figure_save_base_path:
            self.save_figures(figs, figure_save_base_path)
            
        return results, figs