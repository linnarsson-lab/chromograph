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
import multiprocessing as mp

from chromograph.pipeline import config
from chromograph.peak_analysis.utils import *
from chromograph.peak_analysis.feature_selection_by_PearsonResiduals import FeatureSelectionByPearsonResiduals

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

    def fit(self, ds: loompy.LoomConnection, out_file: str, agg_spec: Dict[str, str] = None, reorder: bool = True, motifs: bool = True, by_attr: str = 'Clusters') -> None:
        '''
        Aggregate the matrix, find markers and annotate enriched motifs by homer
        
        Args:
            ds              LoomConnection. Dataset must contain ds.ra.Gene and ds.ca.Clusters
            outfile         Filename of aggregation file
            agg_spec        Dictionary containing numpy-groupies function to be applied to column attributes
        '''
        self.outdir = '/' + os.path.join(*out_file.split('/')[:-1], 'exported')
        self.motifdir = '/' + os.path.join(*out_file.split('/')[:-1], 'motifs')
        self.peakdir = '/' + os.path.join(*out_file.split('/')[:-1], 'peaks')

        if not 'NPeaks' in ds.ca:
            logging.info('Calculating peak and cell coverage')
            ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
            ds.ca['NPeaks'] = ds.map([np.count_nonzero], axis=1)[0]
            ds.ca['FRIP'] = div0(ds.ca.NPeaks, ds.ca.passed_filters)

        agg_spec = {
            "Age": "mean",
            "Clusters": "first",
            "Class": "mode",
            "NPeaks": "mean",
            "Sex": "tally",
            "Tissue": "tally",
            "Chemistry": "tally",
            "SampleID": "tally",
            "TissuePool": "first",
            "Outliers": "mean",
            "Aneuploid": 'mode',
            "PCW": "mean"
        }

        cells = ds.col_attrs[by_attr] >= 0
        labels = ds.col_attrs[by_attr][cells]
        n_labels = len(set(labels))
        name = out_file.split('/')[-1].split('_')[0]

        logging.info(f"Aggregating by {by_attr}")
        ds.aggregate(out_file, None, by_attr, "sum", agg_spec)
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
            dsout['binary'], dsout.ca['CPM_thres'] = KneeBinarization(dsout, bounds=(5,40))

            ## Select cluster markers by residuals
            _ = Enrichment_by_residuals(dsout)
            dsout.ra.Valid2, dsout.ra.residuals = FeatureSelectionByPearsonResiduals(n_genes=self.config.params.N_peaks_decomp, layer='CPM', mask=np.isin(ds.ra.Chr, ['chrX', 'chrY'])).fit(dsout)
            markers = np.where(dsout.ra.Valid2)[0]

            # Renumber the clusters
            if reorder:
                logging.info("Renumbering by similarity, and permuting columns")

                data = dsout[:, :][markers, :].T
                data[np.where(data<0)] = 0  ## BUG handling. Sometimes values surpass the bit limit in malignant cells
                data = np.log(data + 1)
                D = pdist(data, 'correlation')
                Z = hc.linkage(D, 'ward', optimal_ordering=True)
                ordering = hc.leaves_list(Z)

                # Permute the aggregated file, and renumber
                dsout.permute(ordering, axis=1)
                dsout.ca[by_attr] = np.arange(n_labels)

                # Renumber the original file, and permute
                d = dict(zip(ordering, np.arange(n_labels)))
                new_clusters = np.array([d[x] if x in d else -1 for x in ds.ca[by_attr]])
                ds.ca[by_attr] = new_clusters
                ds.permute(np.argsort(ds.col_attrs[by_attr]), axis=1)

            # Redo the Ward's linkage just to get a tree that corresponds with the new ordering
            data = dsout[:, :][markers, :].T
            data[np.where(data<0)] = 0  ## BUG handling. Sometimes values surpass the bit limit in malignant cells
            data = np.log(data + 1)
            D = pdist(data, 'correlation')
            dsout.attrs.linkage = hc.linkage(D, 'ward', optimal_ordering=True)

            if motifs:
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

                if os.path.isfile(os.path.join(self.peakdir, 'Compounded_peaks.bed')):
                    bg_file = os.path.join(self.peakdir, 'Compounded_peaks.bed')
                else:
                    bg_file = os.path.join(self.config.paths.build, 'All', 'peaks', 'Compounded_peaks.bed')
                
                for pile in piles:
                    Homer_find_motifs(bed=pile[0], outdir=pile[1], homer_path=self.config.paths.HOMER, motifs=os.path.join(chromograph.__path__[0], 'references/human_TFs.motifs'), cpus=mp.cpu_count())

                dsout.ca.Enriched_Motifs = retrieve_enrichments(dsout, self.motifdir, N=self.config.params.N_most_enriched)

                ## Plot results 
                if 'UMAP' in ds.ca:
                    logging.info("Plotting UMAP")
                    manifold(ds, os.path.join(self.outdir, f"{name}_peaks_UMAP.png"), list(dsout.ca.Enriched_Motifs), embedding = 'UMAP')
                logging.info("Plotting TSNE")
                manifold(ds, os.path.join(self.outdir, f"{name}_peaks_TSNE.png"), list(dsout.ca.Enriched_Motifs), embedding = 'TSNE')

            ## Plotting neighborhoods and metromap
            cgplot.radius_characteristics(ds, os.path.join(self.outdir, f"{name}_neighborhouds.png"))
            # cgplot.metromap(ds, dsout, os.path.join(self.outdir, f"{name}_metromap.png"), embedding = 'TSNE')

            return
