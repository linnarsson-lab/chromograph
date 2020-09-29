#Imports
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import loompy
from typing import *
import multiprocessing as mp
import pybedtools
from pybedtools import BedTool

from chromograph.pipeline import config
from chromograph.peak_analysis.utils import *

from cytograph.species import Species
from cytograph.annotation import AutoAnnotator, AutoAutoAnnotator
from cytograph.enrichment import FeatureSelectionByMultilevelEnrichment
from cytograph.manifold import GraphSkeletonizer
import cytograph.plotting as cgplot
from cytograph.plotting import manifold

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
        Aggregate the Peak matrix
        '''
        self.config = config.load_config()

    def fit(self, ds: loompy.LoomConnection, out_file: str, agg_spec: Dict[str, str] = None) -> None:
        '''
        Aggregate the matrix, find markers and annotate enriched motifs by homer
        
        Args:
            ds              LoomConnection. Dataset must contain ds.ra.Gene and ds.ca.Clusters
            outfile         Filename of aggregation file
            agg_spec        Dictionary containing numpy-groupies function to be applied to column attributes
        '''
        self.outdir = '/' + os.path.join(*out_file.split('/')[:-1], 'exported')
        self.motifdir = '/' + os.path.join(*out_file.split('/')[:-1], 'motifs')

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

            logging.info('Calculate coverage metrics')
            dsout.ca.Total = dsout.map([np.sum], axis=1)[0]
            dsout.ra.NCells = dsout.map([np.sum], axis=0)[0]
            dsout.ca.NCells = np.bincount(labels, minlength=n_labels)

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
                idx = np.sort(dsout['q_val'][:,i].argsort()[:2000])
                dsout['marker_peaks'][idx,i] = 1
            markers =dsout['marker_peaks'].map([np.sum], axis=0)[0] > 0
            dsout.ra.markerPeaks = markers

            # Renumber the clusters
            logging.info("Renumbering clusters by similarity, and permuting columns")

            data = np.log(dsout[:, :] + 1)[markers, :].T
            D = pdist(data, 'correlation')
            Z = hc.linkage(D, 'ward', optimal_ordering=True)
            ordering = hc.leaves_list(Z)

            # Permute the aggregated file, and renumber
            dsout.permute(ordering, axis=1)
            dsout.ca.Clusters = np.arange(n_labels)

            # Redo the Ward's linkage just to get a tree that corresponds with the new ordering
            data = np.log(dsout[:, :] + 1)[markers, :].T
            D = pdist(data, 'correlation')
            dsout.attrs.linkage = hc.linkage(D, 'ward', optimal_ordering=True)

            # Renumber the original file, and permute
            d = dict(zip(ordering, np.arange(n_labels)))
            new_clusters = np.array([d[x] if x in d else -1 for x in ds.ca.Clusters])
            ds.ca.Clusters = new_clusters
            ds.permute(np.argsort(ds.col_attrs["Clusters"]), axis=1)

            ## Run Homer findMotifs to find the top 5 motifs per cluster
            logging.info(f'Finding enriched motifs among marker peaks')
            
            if not os.path.isdir(self.motifdir):
                os.mkdir(self.motifdir) 

            piles = []
            for x in range(dsout.shape[1]):
                Valids = dsout.layers['marker_peaks'][:,x]
                bed_file = os.path.join(self.motifdir, f'Cluster_{x}.bed')
                peaks = BedTool([(dsout.ra['Chr'][x], str(dsout.ra['Start'][x]), str(dsout.ra['End'][x]), str(dsout.ra['ID'][x]), '.', '+') for x in np.where(Valids)[0]]).saveas(bed_file)
                piles.append([bed_file, os.path.join(self.motifdir, f'Cluster_{x}')])

            # with mp.get_context().Pool(10, maxtasksperchild=1) as pool:
            #     for pile in piles:
            #         pool.apply_async(Homer_find_motifs, args=(pile[0], pile[1], self.config.paths.HOMER, os.path.join(chromograph.__path__[0], 'references/human_TFs.motifs'),))
            #     pool.close()
            #     pool.join()

            for pile in piles:
                Homer_find_motifs(pile[0], pile[1], self.config.paths.HOMER, os.path.join(chromograph.__path__[0], 'references/human_TFs.motifs'), self.config.execution.n_cpus)

            dsout.ca.Enriched_Motifs = retrieve_enrichments(dsout, self.motifdir, N=self.config.params.N_most_enriched)

            # logging.info("Graph skeletonization")
            GraphSkeletonizer(min_pct=1).abstract(ds, dsout)

            ## Plot results 
            name = out_file.split('/')[-1].split('_')[0]
            if 'UMAP' in ds.ca:
                logging.info("Plotting UMAP")
                manifold(ds, os.path.join(self.outdir, f"{name}_peaks_manifold_UMAP.png"), list(dsout.ca.Enriched_Motifs), embedding = 'UMAP')
            logging.info("Plotting TSNE")
            manifold(ds, os.path.join(self.outdir, f"{name}_peaks_manifold_TSNE.png"), list(dsout.ca.Enriched_Motifs), embedding = 'TSNE')

            cgplot.radius_characteristics(ds, os.path.join(self.outdir, f"{name}_All_neighborhouds.png"))
            cgplot.metromap(ds, dsout, os.path.join(self.outdir, f"{name}_metromap.png"), embedding = 'TSNE')

            return
