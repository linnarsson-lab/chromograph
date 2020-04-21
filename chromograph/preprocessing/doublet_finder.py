# This is an alteration by Camiel Mannens of the doublet detection function with the purpose of being applied to scATAC-seq data.
# The original function was written by Kimberly Siletti based on doubletFinder.R as forwarded by the Allen Institute:
#
# "Doublet detection in single-cell RNA sequencing data"
#
# This function generates artificial nearest neighbors from existing single-cell ATAC
# sequencing data. First, real and artificial data are merged. Second, dimension reduction
# is performed on the merged real-artificial dataset using TF-IDF and PCA. Third, the proportion of
# artificial nearest neighbors is defined for each real cell. Finally, real cells are rank-
# ordered and predicted doublets are defined via thresholding based on the expected number
# of doublets.
#
# @param seu A fully-processed Seurat object (i.e. after normalization, variable gene definition,
# scaling, PCA, and tSNE).
# @param expected.doublets The number of doublets expected to be present in the original data.
# This value can best be estimated from cell loading densities into the 10X/Drop-Seq device.
# @param porportion.artificial The proportion (from 0-1) of the merged real-artificial dataset
# that is artificial. In other words, this argument defines the total number of artificial doublets.
# Default is set to 25%, based on optimization on PBMCs (see McGinnis, Murrow and Gartner 2018, BioRxiv).
# @param proportion.NN The proportion (from 0-1) of the merged real-artificial dataset used to define
# each cell's neighborhood in PC space. Default set to 1%, based on optimization on PBMCs (see McGinnis,
# Murrow and Gartner 2018, BioRxiv).
# @return An updated Seurat object with metadata for pANN values and doublet predictions.
# @export
# @examples
# seu <- doubletFinder(seu, expected.doublets = 1000, proportion.artificial = 0.25, proportion.NN = 0.01)"

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import loompy
from scipy import sparse
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from unidip import UniDip
from sklearn.ensemble import IsolationForest

import cytograph as cg
from cytograph.manifold import BalancedKNN
from cytograph.metrics import jensen_shannon_distance
from cytograph.clustering import PolishedLouvain, PolishedSurprise
from cytograph.plotting import manifold, doublets_plots

sys.path.append('/home/camiel/chromograph')
from chromograph.pipeline.TF_IDF import TF_IDF

import logging

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

def doublet_finder(ds: loompy.LoomConnection, proportion_artificial: float = 0.20, fixed_th: float = None, k: int = None, qc_dir: object = ".", max_th: float= 1) -> np.ndarray:
    '''
    '''
    ## Create n doublets
    n_real_cells = ds.shape[1]
    n_doublets = int(n_real_cells / (1 - proportion_artificial) - n_real_cells)    
    fdb = 'doublets.loom' # Filename for temporary loom file containing doublets

    ## Use only Q25 top bins
    logging.info(f'Calculating row wise nonzero rate')
    NCells = ds.map([np.count_nonzero], axis=0)[0]
    q = np.quantile(NCells, .75)
    logging.info(f'Using only bins present in more than {q} out of {ds.shape[1]} cells')
    valid = NCells > q

    ## Subset real data to valid rows and create empty doublet array
    real_data = ds[:,:][valid,:]
    doublets = np.zeros((np.sum(valid), n_doublets))
    
    logging.info('Generating artificial doublets')
    ## Create doublets
    for i in range(n_doublets):
        a = np.random.choice(ds.shape[1])
        b = np.random.choice(ds.shape[1])
        doublets[:, i] = real_data[:, a] + real_data[:, b]
    
    ## Concatenate data and create temporary loom
    A = sparse.coo_matrix(real_data)
    B = sparse.coo_matrix(doublets)
    data = sparse.hstack([A,B])

    logging.info(f'Test data has {data.shape} shape')

    logging.info('Creating temporary loom file')
    cas = {'CellID': np.concatenate([ds.ca['CellID'], np.array(['_'.join(['doublet', str(x)]) for x in range(n_doublets)])]), 'Cell': np.concatenate([[1] * ds.shape[1], [0] * n_doublets])}
    ras = {'loc': ds.ra.loc[valid]}
    loompy.create(filename=fdb, layers=data, col_attrs=cas, row_attrs=ras)

    del data, doublets

    with loompy.connect(fdb, 'r+') as dsb:
        ## normalize data
        logging.info(f'Performing TF-IDF')
        tf_idf = TF_IDF()
        tf_idf.fit(dsb)
        dsb.layers['TF_IDF'] = 'float16'
        for (ix, selection, view) in dsb.scan(axis=1):
            dsb['TF_IDF'][:,selection] = tf_idf.transform(view[:,:], selection)
            logging.info(f'transformed {max(selection)} cells')
        ## Fit PCA
        logging.info('Fitting PCA')
        pca = PCA(n_components=50).fit_transform(dsb['TF_IDF'][:,:].T)
        dsb.ca.PCA = pca
        if k is None:
            k = int(np.min([100, ds.shape[1] * 0.01]))

        logging.info(f"Initialize NN structure with k = {k}")
        knn_result = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
        knn_result.fit(pca)
        knn_dist, knn_idx = knn_result.kneighbors(X=pca, return_distance=True)

        num = ds.shape[1]
        knn_result1 = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=4)
        knn_result1.fit(pca[0:num, :])
        knn_dist1, knn_idx1 = knn_result1.kneighbors(X=pca[num + 1:, :], n_neighbors=10)
        knn_dist_rc, knn_idx_rc = knn_result1.kneighbors(X=pca[0:num, :], return_distance=True)

        logging.info(f'Plot TSNE of data with doublets')
        tsne = TSNE(angle=0.5, perplexity= 30) ## perplexity at 30 because of small cell number
        dsb.ca.TSNE = tsne.fit(dsb.ca.PCA).embedding_

        plt.figure(figsize=(8,8))
        plt.scatter(dsb.ca.TSNE[dsb.ca.Cell == 1, 0], dsb.ca.TSNE[dsb.ca.Cell == 1, 1], c='#8da0cb', s=10)
        plt.scatter(dsb.ca.TSNE[dsb.ca.Cell == 0, 0], dsb.ca.TSNE[dsb.ca.Cell == 0, 1], c='#fc8d62', s=10)
        plt.savefig(os.path.join(qc_dir, 'TSNE_with_doublets.png'))
        
    dist_th = np.mean(knn_dist1.flatten()) + 1.64 * np.std(knn_dist1.flatten())

    doublet_freq = np.logical_and(knn_idx > ds.shape[1], knn_dist < dist_th)
    doublet_freq_A = doublet_freq[ds.shape[1]:ds.shape[1]+n_doublets, :]
    mean1 = doublet_freq_A.mean(axis=1)
    mean2 = doublet_freq_A[:, 0:int(np.ceil(k / 2))].mean(axis=1)
    doublet_score_A = np.maximum(mean1, mean2)

    doublet_freq = doublet_freq[0:ds.shape[1], :]
    mean1 = doublet_freq.mean(axis=1)
    mean2 = doublet_freq[:, 0:int(np.ceil(k / 2))].mean(axis=1)
    doublet_score = np.maximum(mean1, mean2)
    doublet_flag = np.zeros(ds.shape[1],int)
    doublet_th1 = 1
    doublet_th2 = 1
    doublet_th = 1
    #Infer TH from the data or use fixed TH

    # instantiate and fit the KDE model
    kde = KernelDensity(bandwidth=0.1  , kernel='gaussian')
    kde.fit(doublet_score_A[:, None])

    # score_samples returns the log of the probability density
    xx = np.linspace(doublet_score_A.min(), doublet_score_A.max(), len(doublet_score_A)).reshape(-1,1)

    logprob = kde.score_samples(xx)
    if fixed_th is not None:
        doublet_th = float(fixed_th)
        
    else:
        #Check if the distribution is bimodal
        intervals = UniDip(np.exp(logprob)).run()
        if (len(intervals)>1):
            kmeans = KMeans(n_clusters=2).fit(doublet_score_A.reshape(len(doublet_score_A),1))
            high_cluster = np.where(kmeans.cluster_centers_==max(kmeans.cluster_centers_))[0][0]
            doublet_th1 = np.around(np.min(doublet_score_A[kmeans.labels_==high_cluster]),decimals=3)

        #0.5% for every 1000 cells - the rate of detectable doublets by 10X 
        doublet_th2 = np.percentile(doublet_score,100-(5e-4*ds.shape[1]))
        doublet_th2 = np.around(doublet_th2,decimals=3)
        logging.info(f'th1: {doublet_th1} th2: {doublet_th2}')
        #The TH shouldn't be higher than indicated
        if  doublet_th2 >max_th:
            doublet_th2= max_th
        if  doublet_th1 >max_th:
            doublet_th1= max_th
        if (len(np.where(doublet_score>=doublet_th1)[0])>(len(np.where(doublet_score>=doublet_th2)[0]))):
            doublet_th = doublet_th2
        else:
            doublet_th = doublet_th1

    doublet_flag[doublet_score>=doublet_th]=1
    logging.info(f'Doublet threshold is set at {doublet_th}, cells passing threshold: {np.sum(doublet_flag==1)}')
    
    #Calculate the score for the cells that are nn of the marked doublets 
    pca_rc = pca[0:n_real_cells, :]
    knn_dist1_rc, knn_idx1_rc = knn_result1.kneighbors(X=pca_rc[doublet_flag==1,:],n_neighbors=10, return_distance=True)
    
    dist_th = np.mean(knn_dist1_rc.flatten()) + 1.64 * np.std(knn_dist1_rc.flatten())
    doublet2_freq = np.logical_and(doublet_flag[knn_idx_rc]==1  , knn_dist_rc < dist_th)
    doublet2_nn =  knn_dist_rc < dist_th
    doublet2_score  = doublet2_freq.sum(axis=1)/doublet2_nn.sum(axis=1)

    doublet_flag[np.logical_and(doublet_flag == 0 ,doublet2_score >= doublet_th/2)] = 2
    
    ds.ca.PCA = pca[0:ds.shape[1], :]
    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    doublets_plots.doublets_TSNE(ax[0], ds, doublet_flag)
    doublets_plots.fake_doublets_dist(ax[1], doublet_score_A, logprob, xx, doublet_th1, doublet_th2, doublet_th)
    fig.savefig(os.path.join(qc_dir, 'doublet_plot.png'), dpi=144)
    logging.info(f"Doublet fraction: {100*len(np.where(doublet_flag>0)[0])/ds.shape[1]:.2f}%, {len(np.where(doublet_flag>0)[0])} cells. \n\t\t\t(Expected detectable doublet fraction: {(5e-4*ds.shape[1]):.2f}%)")
    
    ## Cleanup
    os.remove(fdb)
    return doublet_score,doublet_flag