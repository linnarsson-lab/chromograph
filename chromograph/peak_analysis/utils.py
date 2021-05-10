#import
import loompy
import os
import sys
import numpy as np
from tqdm import tqdm
from typing import *
import subprocess

import chromograph
from chromograph.pipeline.utils import *

import fisher
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
from kneed import KneeLocator

import numpy as np
import pandas as pd
import os
import sys
import collections
import matplotlib.pyplot as plt
import gzip
import loompy
import scipy.sparse as sparse
import urllib.request
import pybedtools
import warnings
import logging
from tqdm import tqdm
from numba.core.errors import NumbaPerformanceWarning
import multiprocessing as mp
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

from cytograph.decomposition import HPF
from scipy.stats import poisson
from cytograph.manifold import BalancedKNN
from cytograph.metrics import jensen_shannon_distance
from cytograph.embedding import art_of_tsne
from cytograph.clustering import PolishedLouvain, PolishedSurprise
from cytograph.plotting import manifold

from chromograph.plotting.QC_plot import QC_plot
from chromograph.pipeline import config
from chromograph.pipeline.utils import *
from chromograph.pipeline.TF_IDF import TF_IDF
from chromograph.pipeline.PCA import PCA
from chromograph.peak_analysis.feature_selection_by_variance import FeatureSelectionByVariance

from pynndescent import NNDescent
from umap import UMAP
from joblib import parallel_backend
import sklearn.metrics
from scipy.spatial import distance
from harmony import harmonize
import community
import networkx as nx
from scipy import sparse
from typing import *

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
        c[:,0] = np.array(ds[:,np.where(ds.ca.Clusters==label)[0]]).astype('int').flatten()
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

def KneeBinarization(dsagg: loompy.LoomConnection, bins: int = 200, mode: str = 'linear', bounds:tuple=(40,200)):
    '''
    Identifies positive peaks for every cluster based on the decay curve of the CPM values
    
    Args:
        ds          LoomConnection to aggregated peak file
        bins        Number of bins to fit CPM scores into. Defaults to 200. User lower number for small cell counts.
                    High values can cause rougher curves.
        mode        'linear' or 'log'. Determines if values are scaled before determining inflection point. In general
                    'log' is more lenient in feature selection.
    Returns:
        peaks       Numpy array of positive peaks
        CPM_thres   Thresholds used for peak binarization in cluster
    '''
    logging.info(f'Binarize clusters by knee point')
    ## Create layer
    peaks = np.zeros(dsagg.shape)
    CPM_thres = np.zeros(dsagg.shape[1])
    failed = []
    N_pos = []

    for cls in tqdm(dsagg.ca.Clusters):

        if mode == 'linear':
            vals = dsagg['CPM'][:,np.where(dsagg.ca.Clusters==cls)[0]]
            values, base = np.histogram(vals, bins = bins)
            cumulative = np.cumsum(values)

            x = base[:-1]
            y = len(vals)-cumulative

            kn = KneeLocator(x, y, curve='convex', direction='decreasing', interp_method='polynomial')
            t = kn.knee

            if (t > bounds[1]) or (t < bounds[0]):
                failed.append(cls)

            else:
                CPM_thres[dsagg.ca.Clusters==cls] = t
                valid = vals > t
                N_pos.append(np.sum(valid))
                peaks[:,dsagg.ca.Clusters==cls] = valid


        elif mode == 'log':
            vals = np.log10(dsagg['CPM'][:,np.where(dsagg.ca.Clusters==cls)[0]]+1)
            values, base = np.histogram(vals, bins = bins)
            cumulative = np.cumsum(values)

            x = base[:-1]
            y = np.log10((len(vals)-cumulative)+1)

            kn = KneeLocator(x, y, curve='concave', direction='decreasing', interp_method='polynomial')
            t = 10**kn.knee
            
            if t > bounds[1]:
                failed.append(cls)

            else:
                CPM_thres[dsagg.ca.Clusters==cls] = t
                valid = vals > np.log10(t)
                N_pos.append(np.sum(valid))
                peaks[:,dsagg.ca.Clusters==cls] = valid

        else:
            logging.info('No correct mode selected!')
            return
        
    ## Set threshold in accordance with mean number of positive features
    if len(failed) > 0:
        logging.info(f'failed to set threshold in {len(failed)} clusters')
        N_feat = np.mean(N_pos)
        for cls in failed:
            vals = dsagg['CPM'][:,np.where(dsagg.ca.Clusters==cls)[0]]
            t = np.quantile(vals, 1-(N_feat/vals.shape[0]))
            CPM_thres[dsagg.ca.Clusters==cls] = t
            peaks[:,dsagg.ca.Clusters==cls] = vals > t

    return peaks, CPM_thres

def get_conservation_score(ds):
    '''
    Overlays peaks of PeakxCell matrix with phastcon100way reference to estimate estimate evolutionaryconservation.
    
    Args:
        ds        LoomConnection with 'ID', 'Chr', 'Start' and 'End' row attributes
        
    Returns
        out       np.array or shape ds.shape[1] containing conservation scores
    '''
    BedTool([(ds.ra['Chr'][x], str(ds.ra['Start'][x]), str(ds.ra['End'][x]), str(ds.ra['ID'][x])) for x in range(ds.shape[0])]).saveas('input.bed')
    try:
        subprocess.call(['bigWigAverageOverBed', '/data/proj/scATAC/ref/hg38.phastCons100way.bw', 'input.bed', 'out.tab'])
        tab = np.loadtxt('out.tab', dtype=str, delimiter='\t')

    except:
        logging.info(f'Could not find bigWigAverageOverBed on path')
    ## Cleanup
    subprocess.call(['rm', 'input.bed', 'out.tab'])

    return np.array(tab[:,-1].astype('float'))

    
def Homer_find_motifs(bed, outdir, homer_path, motifs, cpus=4):
    """
    Call Homer find motifs to identiy the most enriched motifs per cluster for the top N most enriched peaks.

    args:
        data        list of [clustername, path to BED-file]
        pf          path to peaks directory
        macs_path   path to MACS

    return:
        location of aggregated peaks file
    """
    homer = os.path.join(homer_path, 'findMotifsGenome.pl')

    ## Call Peaks
    subprocess.run([homer, bed, 'hg38', outdir, '-mknown', motifs, '-p', str(cpus), '-nomotif'])
    
    # ## We only need the narrowPeak file, so clean up the rest
    subprocess.run(['rm', bed])
    return f'Completed {outdir}'

def retrieve_enrichments(ds, motif_dir, N=5):
    '''
    Retrieved the top N motifs from Homer findMotifs results
    '''
    ld = os.listdir(motif_dir)
    c_dict = {}
    for d in ld:
        n = int(d.split('_')[-1])
        mat = np.loadtxt(os.path.join(motif_dir, d, 'knownResults.txt'), dtype=str, skiprows=1)
        c_dict[n] = ' '.join([x.split('_')[0] for x in mat[:N,0]])
    
    motif_markers = np.array([c_dict[x] for x in ds.ca.Clusters])
    return motif_markers

class iterativeLSI:
    def __init__(self) -> None:
        '''
        Fit iterative LSI to detect variable peaks
        '''
        self.config = config.load_config()

    def fit(self, ds: loompy.LoomConnection, skip_TFIDF:bool = False):
        '''
        '''
        logging.info(f'Performing Preclustering LSI')
        if not skip_TFIDF == True:
            ds.ra.Valid = ds.ra['NCells'] > np.quantile(ds.ra['NCells'], 1 - (20000/ds.shape[0]))
            logging.info(f'Performing TF-IDF')
            tf_idf = TF_IDF(layer='Binary')
            tf_idf.fit(ds, items=ds.ra.Valid)
            ds.layers['TF-IDF'] = 'float16'
            logging.info(f'Transforming')
            for (_, selection, view) in ds.scan(axis=1):
                ds['TF-IDF'][:,selection] = tf_idf.transform(view['Binary'][:,:], selection)
            del tf_idf
            logging.info(f'Finished fitting TF-IDF')

        ## Fit PCA
        logging.info(f'Fitting PCA to layer TF-IDF')
        pca = PCA(max_n_components = 40, layer= 'TF-IDF', key_depth= 'NPeaks', batch_keys = self.config.params.batch_keys)
        pca.fit(ds)

        ## Decompose data
        ds.ca.LSI = pca.transform(ds)

        logging.info(f'Finished PCA transformation')
        del pca

        ## Get correct embedding and metric
        decomp = ds.ca.LSI
        metric = "euclidean"

        ## Construct nearest-neighbor graph
        logging.info(f"Computing balanced KNN (k = {25}) space using the '{metric}' metric")
        bnn = BalancedKNN(k=25, metric=metric, maxl=2 * 25, sight_k=2 * 25, n_jobs=-1)
        bnn.fit(decomp)
        knn = bnn.kneighbors_graph(mode='distance')
        knn.eliminate_zeros()
        mknn = knn.minimum(knn.transpose())
        # Convert distances to similarities
        logging.info(f'Converting distances to similarities knn shape.')
        max_d = knn.data.max()
        knn.data = (max_d - knn.data) / max_d
        mknn.data = (max_d - mknn.data) / max_d
        ds.col_graphs.KNN = knn
        ds.col_graphs.MKNN = mknn
        mknn = mknn.tocoo()
        mknn.setdiag(0)
        # Compute the effective resolution
        logging.info(f'Computing resolution')
        d = 1 - knn.data
        radius = np.percentile(d, 90)
        logging.info(f"  90th percentile radius: {radius:.02}")
        ds.attrs.radius = radius
        inside = mknn.data > 1 - radius
        rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)
        ds.col_graphs.RNN = rnn

        del knn, mknn, rnn

        ## Perform tSNE
        metric_f = (jensen_shannon_distance if metric == "js" else metric)  # Replace js with the actual function, since OpenTSNE doesn't understand js

        logging.info(f"Computing 2D and 3D embeddings from latent space")
        logging.info(f"Art of tSNE with distance metrid: {metric_f}")
        ds.ca.TSNE = np.array(art_of_tsne(decomp, metric=metric_f))  # art_of_tsne returns a TSNEEmbedding, which can be cast to an ndarray (its actually just a subclass)

        ## Perform Clustering
        logging.info("Performing Polished Louvain clustering")
        pl = PolishedLouvain(outliers=False, graph="RNN", embedding="TSNE", resolution = 1, min_cells=50)
        labels = pl.fit_predict(ds)
        ds.ca.ClustersModularity = labels + min(labels)
        ds.ca.OutliersModularity = (labels == -1).astype('int')
        ds.ca.preClusters = labels + min(labels)
        ds.ca.Outliers = (labels == -1).astype('int')
        logging.info(f"Found {ds.ca.preClusters.max() + 1} clusters")

def select_preclusters(ds, min_cells=50, min_clusters=25):
    '''
    '''
    cnt = Counter(ds.ca.preClusters)
    
    ## Get Cells per cluster
    vals = np.array([v for k,v in cnt.items()])
    N_clusters = np.sum(vals>min_cells)
    perform_iter = N_clusters < min_clusters
    
    logging.info(f'Perform iterative LSI: {perform_iter}, valid clusters: {N_clusters}, required: {min_clusters}')
    
    if perform_iter:
        logging.info(f'Perform Itererative LSI')
        iLSI  = iterativeLSI()
        iLSI.fit(ds)
        valid_clusters = np.unique(ds.ca.preClusters)
    else:
        valid_clusters = [k for k,v in cnt.items() if v >= min_cells]
        
    return set(valid_clusters)