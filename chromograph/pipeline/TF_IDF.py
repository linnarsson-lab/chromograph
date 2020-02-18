import numpy as np

import loompy

from .utils import div0

class TF_IDF:
    """
    Calculate the Term-Frequency Inverse Data Frequency (TF-IDF) values, dealing properly 
    with edge cases such as division by zero.
    """
    def __init__(self, layer: str = "") -> None:
        self.IDF = None  # type: np.ndarray
        self.totals = None  # type: np.ndarray
        self.layer = layer
        self.level = 0

    def fit(self, ds: loompy.LoomConnection) -> None:
        self.IDF = np.zeros(ds.shape[0])  ## Row totals
        self.totals = np.zeros(ds.shape[1])  ## Column totals
        N = ds.shape[1]

        ## Scan over rows (all cells) and add to column totals
        for _, selection, view in ds.scan(axis=0):
            vals = view[self.layer][:, :].astype("float")
            self.totals += np.sum(vals, axis=0)
            
            ## Set row totals
            NB = np.sum(view[self.layer][:, :]>0, axis=1)
            self.IDF[selection] = np.log10(div0(N,NB)+1)
            
        ## Set level to normalize to
        self.level = np.median(self.totals)


    def transform(self, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
        """
        Calculate the TF-IDF score for array using the previously calculated aggregate statistics
        Args:
            vals (ndarray):		Matrix of shape (n_genes, n_cells)
            cells (ndarray):	Optional indices of the cells that are represented in vals
        Returns:
            vals_adjusted (ndarray):	The normalized values
        """
        # Compute the Term-Frequency for selected cells
        if cells is None:
            cells = slice(None)
        vals = vals.astype("float")
        vals = div0(vals, self.totals[cells]) * self.level
        
        ## Multiply by Inverse Data Frequency
        vals = vals*self.IDF[:,None]
        
        return vals

    def fit_transform(self, vals: np.ndarray, cells: np.ndarray = None) -> np.ndarray:
        self.fit(ds)
        return self.transform(vals, cells)