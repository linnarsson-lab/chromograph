#Imports
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import loompy

from chromograph.pipeline import config

import loompy
from cytograph.species import Species
from cytograph.annotation import AutoAnnotator, AutoAutoAnnotator
from cytograph.enrichment import FeatureSelectionByMultilevelEnrichment
from cytograph.manifold import GraphSkeletonizer
import cytograph.plotting as cgplot

from typing import *

import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class GA_Aggregator:
    def __init__(self, *, f: Union[float, List[float]] = 0.5) -> None:
        '''
        Aggregate the Gene-Accessibility signal, find markers and call the auto-annotater
        '''
        self.f = f
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

        if agg_spec is None:
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
        ds.aggregate(out_file, None, "Clusters", "mean", agg_spec)
        with loompy.connect(out_file) as dsout:

            if n_labels <= 1:
                return

            logging.info("Computing cluster gene enrichment scores")
            self.mask = Species.detect(ds).mask(dsout, ("cellcycle", "sex", "ieg", "mt"))
            fe = FeatureSelectionByMultilevelEnrichment(mask=self.mask)
            markers = fe.fit(ds)
            dsout.layers["enrichment"] = fe.enrichment

            ## Save top N most enriched genes
            Most_enriched = [dsout.ra.Gene[dsout['enrichment'][:,i].argsort()[::-1][:self.config.params.N_most_enriched]].tolist() for i in range(dsout.shape[1])]
            dsout.ca.Most_enriched = [" ".join(enr) for enr in Most_enriched]
            dsout.ca.NCells = np.bincount(labels, minlength=n_labels)

            # Reorder the genes, markers first, ordered by enrichment in clusters
            logging.info("Permuting rows")
            mask = np.zeros(ds.shape[0], dtype=bool)
            mask[markers] = 1
            # fetch enrichment from the aggregated file, so we get it already permuted on the column axis
            gene_order = np.zeros(ds.shape[0], dtype='int')
            gene_order[mask] = np.argmax(dsout.layer["enrichment"][np.where(mask)[0], :], axis=1)
            gene_order[~mask] = np.argmax(dsout.layer["enrichment"][np.where(~mask)[0], :], axis=1) + dsout.shape[1]
            gene_order = np.argsort(gene_order)
            ds.permute(gene_order, axis=0)
            dsout.permute(gene_order, axis=0)

            if n_labels > 300:
                dsout.ca.MarkerGenes = np.empty(n_labels, dtype='str')
                dsout.ca.AutoAnnotation = np.empty(n_labels, dtype='str')
                return

            if self.config.params.autoannotater == True:
                logging.info("Computing auto-annotation")
                AutoAnnotator(root=self.config.paths.autoannotation, ds=dsout).annotate(dsout)

                logging.info("Computing auto-auto-annotation")
                AutoAutoAnnotator(n_genes=6).annotate(dsout)

            ## Update Manifolds with markers
            name = out_file.split('/')[-1].split('_')[0]
            if 'UMAP' in ds.ca:
                logging.info("Plotting UMAP")
                cgplot.manifold(ds, os.path.join(self.outdir, f"{name}_manifold_markers_UMAP.png"), list(dsout.ca.Most_enriched), embedding = 'UMAP')
            logging.info("Plotting TSNE")
            cgplot.manifold(ds, os.path.join(self.outdir, f"{name}_manifold_markers_TSNE.png"), list(dsout.ca.Most_enriched), embedding = 'TSNE')

            ## Other gene related plots
            logging.info(f'Plotting heatmaps')
            cgplot.TF_heatmap(ds, dsout, os.path.join(self.outdir, f"{name}_TFs_heatmap.pdf"), layer="")
            cgplot.markerheatmap(ds, dsout, os.path.join(self.outdir, f"{name}_markers_heatmap.pdf"), layer="")
            if "pooled" in ds.layers:
                cgplot.TF_heatmap(ds, dsout, os.path.join(self.outdir, f"{name}_TFs_heatmap_pooled.pdf"), layer="pooled")
                cgplot.markerheatmap(ds, dsout, os.path.join(self.outdir, f"{name}_markers_heatmap_pooled.pdf"), layer="pooled")
