import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import sqlite3 as sqlite

# import cytograph.plotting as cgplot
import loompy

from .config import load_config


class Workflow:
        """
        Shared workflow for every task, implementing cytograph, aggregation and plotting
        Subclasses implement workflows that vary by the way cells are collected
        """
#     def __init__(self, deck: PunchcardDeck, name: str) -> None:
    def __init__(self, name: str) -> None:

        self.config = load_config()
        self.name = name
        self.loom_file = os.path.join(self.config.paths.build, "data", name + ".loom")
        self.export_dir = os.path.join(self.config.paths.build, "exported", name)

    def collect_samples(self) -> None:
        ## WORK ON THIS
        files = [os.path.join(path, '10X' + sample, '10X' + sample + f"_{bsize}.loom") for sample in samples]
        loompy.combine_faster(files, out_file, {}, selections, skip_attrs=["_X", "_Y", "Clusters"])

    
    def process(self) -> None:
        
        # STEP 1: build the .loom file and perform primary manifold learning (Bin Analysis)
        if os.path.exists(self.loom_file):
            logging.info(f"Skipping '{self.name}.loom' because it was already complete.")
        else:
            with Tempname(self.loom_file) as out_file:
                self.collect_cells(out_file)
                with loompy.connect(out_file) as ds:
                    ds.attrs.config = config.to_string()
                    logging.info(f"Collected {ds.shape[1]} cells")
                    Cytograph(config=self.config).fit(ds)

#         # STEP 2: aggregate and create the .agg.loom file
#         if os.path.exists(self.agg_file):
#             logging.info(f"Skipping '{self.name}.agg.loom' because it was already complete.")
#         else:
#             with loompy.connect(self.loom_file) as dsout:
#                 clusts, labels = np.unique(dsout.ca.Clusters, return_inverse=True)
#                 if len(np.unique(clusts)) != dsout.ca.Clusters.max() + 1:
#                     logging.info(f"Renumbering clusters before aggregating.")
#                     dsout.ca.ClustersCollected = dsout.ca.Clusters
#                     dsout.ca.Clusters = labels
#                 with Tempname(self.agg_file) as out_file:
#                     Aggregator(mask=Species.detect(dsout).mask(dsout, config.params.mask)).aggregate(dsout, out_file=out_file)
#                 with loompy.connect(self.agg_file) as dsagg:
#                     dsagg.attrs.config = config.to_string()

#         # STEP 3: export plots
#         if os.path.exists(self.export_dir):
#             logging.info(f"Skipping 'exported/{self.name}' because it was already complete.")
#         else:
#             pool = self.name
#             logging.info(f"Exporting plots for {pool}")
#             with Tempname(self.export_dir) as out_dir:
#                 os.mkdir(out_dir)
#                 with loompy.connect(self.loom_file) as ds:
#                     with loompy.connect(self.agg_file) as dsagg:
#                         cgplot.manifold(ds, os.path.join(out_dir, f"{pool}_TSNE_manifold.png"), list(dsagg.ca.MarkerGenes), list(dsagg.ca.AutoAnnotation))
#                         if "UMAP" in ds.ca:
#                             cgplot.manifold(ds, os.path.join(out_dir, pool + "_UMAP_manifold.png"), list(dsagg.ca.MarkerGenes), list(dsagg.ca.AutoAnnotation), embedding="UMAP")
#                         cgplot.markerheatmap(ds, dsagg, out_file=os.path.join(out_dir, pool + "_markers_pooled_heatmap.pdf"), layer="pooled")
#                         cgplot.markerheatmap(ds, dsagg, out_file=os.path.join(out_dir, pool + "_markers_heatmap.pdf"), layer="")
#                         if "HPF" in ds.ca:
#                             cgplot.factors(ds, base_name=os.path.join(out_dir, pool + "_factors"))
#                         if "CellCycle_G1" in ds.ca:
#                             cgplot.cell_cycle(ds, os.path.join(out_dir, pool + "_cellcycle.png"))
#                         if "KNN" in ds.col_graphs:
#                             cgplot.radius_characteristics(ds, out_file=os.path.join(out_dir, pool + "_neighborhoods.png"))
#                         cgplot.batch_covariates(ds, out_file=os.path.join(out_dir, pool + "_batches.png"))
#                         cgplot.umi_genes(ds, out_file=os.path.join(out_dir, pool + "_umi_genes.png"))
#                         if "velocity" in self.config.steps:
#                             cgplot.embedded_velocity(ds, out_file=os.path.join(out_dir, f"{pool}_velocity.png"))
#                         cgplot.TF_heatmap(ds, dsagg, out_file=os.path.join(out_dir, f"{pool}_TFs_pooled_heatmap.pdf"), layer="pooled")
#                         cgplot.TF_heatmap(ds, dsagg, out_file=os.path.join(out_dir, f"{pool}_TFs_heatmap.pdf"), layer="")
#                         if "GA" in dsagg.col_graphs:
#                             cgplot.metromap(ds, dsagg, out_file=os.path.join(out_dir, f"{pool}_metromap.png"))
#                         if "cluster-validation" in self.config.steps:
#                             ClusterValidator().fit(ds, os.path.join(out_dir, f"{pool}_cluster_pp.png"))

#         # If there's a punchcard for this subset, go ahead and compute the subsets for that card
#         card_for_subset = self.deck.get_card(self.name)
#         if card_for_subset is not None:
#             self.compute_subsets(card_for_subset)
        logging.info("Done.")