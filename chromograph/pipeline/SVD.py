import logging
import sys
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr, ks_2samp
from harmony import harmonize
from tqdm import tqdm

import loompy

from chromograph.pipeline import config

class SVD:
    """
    Project a dataset into a reduced feature space using SVD. The projection can be fit
    to one dataset then used to project another. To work properly, both datasets must be normalized in the same
    way prior to projection.
    """
    def __init__(self, max_n_components: int = 50, layer: str = None, test_significance: bool = True, key_depth: str = None, batch_keys: List[str] = None) -> None:
        """
        Args:
            genes:				The genes to use for the projection
            max_n_components: 	The maximum number of projected components
            layer:				The layer to use as input
            test_significance:	If true, return only a subset of up to max_n_components that are significant
            batch_keys:			Keys (attribute names) to use as batch keys for batch correction, or None to omit batch correction
        """
        self.n_components = max_n_components
        self.test_significance = test_significance
        self.key_depth = key_depth
        self.layer = layer
        self.SVD = None  # type: TruncatedSVD
        self.sigs = None  # type: np.ndarray
        self.batch_keys = batch_keys
        self.config = config.load_config()

    def fit(self, ds: loompy.LoomConnection) -> None:
        '''
        Fit SVD to dataset
        '''
        self.SVD = TruncatedSVD(n_components=self.n_components)
        logging.info(f'Fitting {sum(ds.ra.Valid)} features from {ds.shape[1]} cells to {self.n_components} components')
        self.SVD.fit(ds.layers['TF-IDF'].sparse(rows=np.where(ds.ra.Valid)[0]).T)

    def transform(self, ds: loompy.LoomConnection) -> np.ndarray:
        '''
        Transforming dataset. Must contain row attribute Valid
        '''
        logging.info(f'Transforming data')
        transformed = np.zeros((ds.shape[1], self.n_components))

        progress = tqdm(total=ds.shape[1])
        for (ix, selection, view) in ds.scan(axis=1, batch_size=self.config.params.batch_size):
            transformed[selection, :] = self.SVD.transform(view[self.layer][ds.ra.Valid==1,:].T)
            progress.update(self.config.params.batch_size)
        progress.close()

        ## Test significance
        if self.test_significance:
            # Must select significant components only once, and reuse for future transformations
            if self.sigs is None:
                pvalue_KS = np.zeros(transformed.shape[1])  # pvalue of each component
                for i in range(1, transformed.shape[1]):
                    (_, pvalue_KS[i]) = ks_2samp(transformed[:, i - 1], transformed[:, i])
                self.sigs = np.where(pvalue_KS < 0.1)[0]
                logging.info(f'Found {self.sigs.shape[0]} significant components')

        ## Test depth correlation
        if self.key_depth is not None:
            if self.sigs is None:
                self.sigs = np.fromiter(range(transformed.shape[1]), 'int')
            for i in range(transformed.shape[1]):
                r = pearsonr(ds.ca[self.key_depth], transformed[:,i])
                if abs(r[0]) > 0.6 and r[1] < 0.05:
                    if len(np.where(self.sigs == i)[0])>0:
                        self.sigs = np.delete(self.sigs, np.where(self.sigs == i)[0])
            logging.info(f'Components after discarding depth correlates: {self.sigs.shape[0]}')

        if self.test_significance or self.key_depth is not None:
            transformed = transformed[:, self.sigs]

        if self.batch_keys is not None and len(self.batch_keys) > 0:
            logging.info(f'Running Harmony, batch keys: {self.batch_keys}')
            keys_df = pd.DataFrame.from_dict({k: ds.ca[k] for k in self.batch_keys})
            transformed = harmonize(transformed, keys_df, batch_key=self.batch_keys, n_jobs_kmeans=1)

        return transformed

    def fit_transform(self, ds: loompy.LoomConnection) -> np.ndarray:
        self.fit(ds)
        return self.transform(ds)