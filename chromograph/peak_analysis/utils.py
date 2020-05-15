#import
import loompy
import sys
import numpy as np
from tqdm import tqdm
from typing import *

sys.path.append('/home/camiel/chromograph/')
import chromograph
from chromograph.pipeline.utils import *

import fisher
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
from kneed import KneeLocator


import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

def FisherDifferentialPeaks(ds: loompy.LoomConnection, sig_thres: float = 0.05, mtc_method: str = 'fdr_bh'):
    '''
    Performs Fisher-exact test to identify differentially accessible peaks via one-versus all testing.
    
    Args:
        ds             LoomConnection (aggregated peak file)
        sig_thres      Adjusted significance threshold
        mtc_method     Multiple test correction method. Accepts all methods from statsmodels multipletests
       
    Returns
        enrichment     Numpy array containing odds ratios
        q-values       Numpy array containing adjusted p-values
    '''
    enrichment = np.zeros(ds.shape)
    q_values = np.zeros(ds.shape)
    Total = np.sum(ds.ca.NCells)
    
    logging.info(f'Performing Fisher exact tests')
    for label in tqdm(ds.ca.Clusters):
        n_cells = ds.ca.NCells[ds.ca.Clusters == label]

        c = np.zeros((ds.shape[0],4))
        c[:,0] = np.array(ds[:,ds.ca.Clusters==label]).astype('int').flatten()
        c[:,1] = ds.ra.NCells - c[:,0]
        c[:,2] = n_cells - c[:,0]
        c[:,3] = Total - n_cells - c[:,1]
        c = c.astype(np.uint)

        _, p, _ = fisher.pvalue_npy(c[:, 0], c[:, 1], c[:, 2], c[:, 3])
        odds = div0(c[:, 0] * c[:, 3], c[:, 1] * c[:, 2])

        _ , q, _, _ =  multipletests(p, sig_thres, method=mtc_method)

        enrichment[:,ds.ca.Clusters == label] = odds.reshape((ds.shape[0],1))
        q_values[:,ds.ca.Clusters == label] = np.array(q).reshape((ds.shape[0],1))

    return enrichment, q_values

def KneeBinarization(dsagg: loompy.LoomConnection, bins: int = 200):
    '''
    Identifies positive peaks for every cluster based on the decay curve of the CPM values
    
    Args:
        ds        LoomConnection to aggregated peak file
        bins      Number of bins to fit CPM scores into. Defaults to 200. User lower number for small cell counts.
                  High values can cause rougher curves.
    Returns:
        peaks       Numpy array of positive peaks
        CPM_thres   Thresholds used for peak binarization in cluster
    '''
    logging.info(f'Binarize peak matrix')
    ## Create layer
    peaks = np.zeros(dsagg.shape)
    CPM_thres = np.zeros(dsagg.shape[1])

    for cls in tqdm(dsagg.ca.Clusters):
        vals = dsagg['CPM'][:,dsagg.ca.Clusters==cls]
        values, base = np.histogram(vals, bins = bins)
        cumulative = np.cumsum(values)

        x = base[:-1]
        y = len(vals)-cumulative

        kn = KneeLocator(x, y, curve='convex', direction='decreasing', interp_method='polynomial')
 
        CPM_thres[dsagg.ca.Clusters==cls] = kn.knee
        valid = vals > kn.knee
        peaks[:,dsagg.ca.Clusters==cls] = valid
    return peaks, CPM_thres