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
                t = 10**np.quantile(vals,1-(5000/vals.shape[0]))

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
