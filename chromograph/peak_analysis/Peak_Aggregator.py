#Imports
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import loompy

sys.path.append('/home/camiel/chromograph/')
from chromograph.pipeline import config
from chromograph.peak_analysis import *

import loompy
from typing import *

import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class Peak_Aggregator:
    def __init__(self) -> None:
        '''
        Aggregate the Gene-Accessibility signal, find markers and call the auto-annotater
        '''
        self.config = config.load_config() # Generic config, just to get the paths

    def fit(self, ds: loompy.LoomConnection, out_file: str, agg_spec: Dict[str, str] = None) -> None:
        '''
        Aggregate the Gene-Accessibility signal, find markers and call the auto-annotater
        
        Args:
            ds              LoomConnection. Dataset must contain ds.ra.Gene and ds.ca.Clusters
            outfile         Filename of aggregation file
            agg_spec        Dictionary containing numpy-groupies function to be applied to column attributes
        '''
        self.outdir = '/' + os.path.join(*out_file.split('/')[:-1], 'exported')

        agg_spec = {
            "Age": "tally",
            "Clusters": "first",
            "Class": "mode",
            "Total": "mean",
            "Sex": "tally",
            "Tissue": "tally",
            "SampleID": "tally",
            "TissuePool": "first",
            "Outliers": "mean",
            "PCW": "mean"
        }
        cells = ds.col_attrs["Clusters"] >= 0
        labels = ds.col_attrs["Clusters"][cells]
        n_labels = len(set(labels))

        logging.info("Aggregating clusters")
        ds.aggregate(out_file, None, "Clusters", "sum", agg_spec)
        with loompy.connect(out_file) as dsout:

            if n_labels <= 1:
                return

            logging.info('Calculate coverag metrics')
            dsout.ca.NCells = np.bincount(labels, minlength=n_labels)
            dsout.ca.Total = dsout.map([np.sum], axis=1)[0]
            dsout.ra.NCells = dsout.map([np.sum], axis=0)[0]

            ## Normalize peak counts by total fragments per cluster
            logging.info('Convert to CPMs')
            dsout.layers['CPM'] = div0(dsout[''][:,:], dsout.ca.Total * 1e-6)

            ## Call positive and negative peaks for every cluster
            dsout['binary'], dsout.ca['CPM_thres'] = KneeBinarization(dsout)
            
            ## Perform fisher exact for peak counts
            dsout['enrichment'], dsout['q_val'] = FisherDifferentialPeaks(dsout)

            ## Select top N enriched peaks per cluster by odss-ratio
            dsout['marker_peaks'] = 'int8'
            for i in range(dsout.shape[1]):
                idx = np.sort(dsout['enrichment'][:,i].argsort()[::-1][:1000])
                dsout['marker_peaks'][idx,i] = 1
            dsout.ra.markerPeaks = dsout.map([np.nonzero], axis=0)[0] > 0
