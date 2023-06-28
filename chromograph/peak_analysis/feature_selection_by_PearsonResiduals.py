import numpy as np
import loompy
import logging
from chromograph.pipeline.utils import div0

class FeatureSelectionByPearsonResiduals:
    def __init__(self, n_genes: int, mask: np.ndarray = None) -> None:
        self.n_genes = n_genes
        self.mask = mask

    def fit(self, ds: loompy.LoomConnection, theta:int=100) -> np.ndarray:
        """
        Fits a noise model (CV vs mean)
        Args:
            ds (LoomConnection):	Dataset
        Returns:
            ndarray of selected genes (bool array)

        Remarks:
            If the row attribute "Valid" exists, only Valid == 1 genes will be selected
        """
        logging.info(f'Calculating residuals')
        if not 'totals' in ds.ca:
            ds.ca.totals = ds.map([np.sum], axis=1)[0]
        if not 'totals' in ds.ra:
            ds.ra.totals = ds.map([np.sum], axis=0)[0]

        Overall_total = np.sum(ds.ca.totals)
    
    
        if "Valid" in ds.ra:
            valid = ds.ra.Valid == 1
            logging.info(f'Picking from {np.sum(valid)} features')
        else:
            valid = np.ones(ds.shape[0], dtype='bool')
            logging.info(f'No list of valid features given. Use all minus masked chromosomes.')
        if self.mask is not None:
            valid = np.logical_and(valid, np.logical_not(self.mask))
        valid = valid.astype('int')

        data = ds[:,:]

        expected = ds.ca.totals[:,None] @ ds.ra.totals[None,:] / Overall_total
        expected = expected.T

        residuals = div0((data-expected), np.sqrt(expected + np.power(expected, 2)/theta))
                
        d = np.var(residuals, axis=1)

        temp = []
        for gene in np.argsort(-d):
            if valid[gene]:
                temp.append(gene)
            if len(temp) >= self.n_genes:
                break

        out = np.zeros(ds.shape[0])
        out[temp] = True
        
        return out, d


    def select(self, ds: loompy.LoomConnection) -> np.ndarray:
        selected = self.fit(ds)
        ds.ra.Selected = selected.astype("int")
        return selected