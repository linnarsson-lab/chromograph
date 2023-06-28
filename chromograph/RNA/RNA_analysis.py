## Imports
import os
import sys
import collections
import loompy
from tqdm import tqdm
from typing import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist

from collections import Counter
from pynndescent import NNDescent

from chromograph.peak_calling.utils import *
from chromograph.pipeline.utils import *
from chromograph.pipeline import config
from chromograph.RNA.utils import *
from chromograph.RNA.FeatureSelectionByMultilevelEnrichment import FeatureSelectionByMultilevelEnrichment
from chromograph.plotting.sample_distribution_plot import sample_distribution_plot

import cytograph as cg
import cytograph.plotting as cgplot
from cytograph.species import Species
from cytograph.annotation import AutoAnnotator, AutoAutoAnnotator
from cytograph.enrichment import Trinarizer, Enrichment
from cytograph.visualization.plot_overview import PlotOverview

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class RNA_analysis():
    def __init__(self, ds, outdir):
        '''
        '''
        self.peak_file = ds.filename
        self.name = ds.filename.split('/')[-2]
        self.RNA_file = '/' + os.path.join(*ds.filename.split('/')[:-1], f'{self.name}_RNA.loom')
        self.RNA_agg = '/' + os.path.join(*ds.filename.split('/')[:-1], f'{self.name}_RNA.agg.loom')
        self.Imputed_file = '/' + os.path.join(*ds.filename.split('/')[:-1], f'{self.name}_RNA_imputed.loom')
        self.peak_agg = '/' + os.path.join(*ds.filename.split('/')[:-1], f'{self.name}_peaks.agg.loom')
        self.config = config.load_config()
        self.outdir = os.path.join(outdir, 'exported')
        
    def generate_RNA_file(self, RNA_files_dir):
        '''
        '''
        logging.info(f'Starting RNA analysis')
        with loompy.connect(self.peak_file) as ds:
            samples = np.unique(ds.ca.Name[np.where(ds.ca.Chemistry == 'multiome_atac')[0]])
            inputfiles = [os.path.join(RNA_files_dir, f"{sample}.loom") for sample in samples]

            selections = []
            for sample, file in zip(samples, inputfiles):
                valid_cells = set(ds.ca.CellID)
                ndigit = len(ds.ca.CellID[0].split(':')[0].split('_'))
                ## Get cells passing filters
                with loompy.connect(file, 'r') as ds2:
                    if not np.sum(np.isin(ds2.ca.CellID, ds.ca.CellID)) > 1:
                        barcodes = rna_barcodes_to_atac(ds2, n=ndigit)
                        if len(ds.ca.CellID[0].split('/')[-1].split('-')) > 1:
                            barcodes = [x + '-1' for x in barcodes]
                    else:
                        barcodes = ds2.ca.CellID
                    good_cells = np.array([x in valid_cells for x in barcodes])
                    selections.append(good_cells)

            if os.path.isfile(self.RNA_file):
                os.remove(self.RNA_file)
            logging.info(f'Combining files')
            loompy.combine_faster(inputfiles, self.RNA_file, selections=selections, key='Accession')

            ## transcribe cell IDs
            with loompy.connect(self.RNA_file) as dsout:
                dsout.ca.RNA_IDs = dsout.ca.CellID
                if not np.sum(np.isin(dsout.ca.CellID, ds.ca.CellID)) > 1:
                    dsout.ca.CellID = rna_barcodes_to_atac(dsout,n=ndigit)

                match = {k:v for v, k in enumerate(ds.ca.CellID)}
                if (len(ds.ca['CellID'][0].split('-'))> 1) & (len(dsout.ca['CellID'][0].split('-'))==0):
                    new_order = np.array([match[x + '-1'] for x in dsout.ca['CellID']])
                else:
                    new_order = np.array([match[x] for x in dsout.ca['CellID']])

                for k in ds.ca:
                    if k != 'CellID':
                        dsout.ca[k] = ds.ca[k][new_order]
                
                if self.name == 'All':
                    sample_distribution_plot(dsout, os.path.join(self.outdir, f"{self.name}_RNA_cell_counts.png"))

                logging.info('ordering file')
                dsout.permute(np.argsort(dsout.col_attrs["Clusters"]), axis=1)

            logging.info(f'Finished creating file')  
            
        
    def Impute_RNA(self):
        '''
        '''
        
        if not os.path.isfile(self.RNA_file):
            logging.info(f'Generate the merged RNA file first!')
            return
        
        logging.info(f'Imputing RNA layer')
        
        with loompy.connect(self.peak_file) as ds:
            with loompy.connect(self.RNA_file) as dsr:
                if ds.shape[1] == dsr.shape[1]:
                    logging.info(f'Imputation not necessary, all cells have RNA measurements')
                    return

        ## Generate Imputation file
        if os.path.isfile(self.Imputed_file):
            os.remove(self.Imputed_file)

        with loompy.connect(self.peak_file) as ds:
            with loompy.connect(self.RNA_file) as dsr:
                RNA_barcodes = set(dsr.ca.CellID)
                x = np.array([x not in RNA_barcodes for x in ds.ca.CellID])
                cells = np.where(x)[0]
                M = dsr.shape[0]
                N = ds.shape[1]

                RNA_barcodes = set(dsr.ca.CellID)

                if len(ds.ca['CellID'][0].split('-'))> 1:
                    x = [x.split('-')[0] not in RNA_barcodes for x in ds.ca.CellID]
                    new_barcodes = ds.ca.CellID[x]
                    rna_bars = [x + '-1' for x in dsr.ca.CellID]
                    all_barcodes = np.concatenate([rna_bars, new_barcodes])
                else:
                    x = [x not in RNA_barcodes for x in ds.ca.CellID]
                    new_barcodes = ds.ca.CellID[x]
                    all_barcodes = np.concatenate([dsr.ca.CellID, new_barcodes])

                if ds.shape[1] > 10000:
                    batch = 1024
                    M = dsr.shape[0]
                    N = batch
                    empty_mat = sparse.csr_matrix((M,N), dtype=dsr[''].dtype)
                    cols = {'CellID': all_barcodes[:batch]}
                    logging.info(f'Create file')
                    loompy.create(self.Imputed_file, empty_mat, dsr.ra, cols)

                    with loompy.connect(self.Imputed_file) as dsout:
                        n_iter = np.ceil(ds.shape[1] / batch).astype(int)

                        pbar = tqdm(len(all_barcodes))
                        for i in range(1,n_iter):
                            start = i * batch
                            end = (i+1) * batch
                            if end > ds.shape[1]:
                                end = ds.shape[1]

                            N = end - start
                            empty_mat = np.zeros((M,N), dtype=dsr[''].dtype)
                            dsout.add_columns({'': empty_mat}, col_attrs={'CellID': all_barcodes[start:end]}, row_attrs=ds.ra)
                            pbar.update(batch)
                        pbar.close()
                 
                else:
                    empty_mat = sparse.csr_matrix((M,N), dtype=dsr[''].dtype)
                    logging.info(f'Creating file')
                    loompy.create(self.Imputed_file, empty_mat, dsr.ra, {'CellID': all_barcodes})

                with loompy.connect(self.Imputed_file) as dsi:
                    logging.info(dsi.shape)
                    logging.info(f'Transferring data and normalizing')

                    ## Tranfer RNA data
                    for layer in dsr.layers:
                        if layer not in dsi.layers:
                            dsi[layer] = "float32"
                        s = dsr[layer].shape[1]
                        for (_, indexes, view) in tqdm(dsr.scan(axis=0)):
                            dsi[layer][indexes,:s] = div0(view[layer][:,:], dsr.ca.TotalUMI) * 5000

                    ## Transfer column attributes
                    logging.info(f'Transfer attributes')
                    transfer_ca(ds, dsi, 'CellID')

        with loompy.connect(self.RNA_file) as dsr:
            with loompy.connect(self.Imputed_file) as dsi:

                if len(np.where(dsi.ca.Chemistry!='multiome_atac')[0]) > 0:
                    logging.info(f'Generating anchor net')
                    anchors = np.where(dsi.ca.Chemistry=='multiome_atac')[0]
                    id_to_anchor = {i: a for i,a in enumerate(anchors)}
                    queries = np.where(dsi.ca.Chemistry!='multiome_atac')[0]
                    id_to_query = {i: a for i,a in enumerate(queries)}
                    index = NNDescent(dsi.ca.LSI[anchors])

                    X = index.query(dsi.ca.LSI[queries],10)

                    data = X[1]
                    max_d = np.max(data)
                    data = (max_d - data) / max_d

                    new_pos = []
                    new_origin = []
                    for i, row in enumerate(X[0]):
                        new_pos.append([id_to_anchor[x] for x in row])
                        new_origin.append([id_to_query[i] for x in range(len(row))])
                    new_pos = np.array(new_pos)   
                    new_origin = np.array(new_origin)

                    nn = sparse.csr_matrix((data.flatten(), (new_origin.flatten(),new_pos.flatten())), shape=(dsi.shape[1],dsi.shape[1]), dtype='float')
                    nn.eliminate_zeros()
                    nn[anchors,anchors] = 1
                    dsi.col_graphs['anchor_net'] = nn

                    total_link = np.asarray(div0(1, np.sum(nn, axis=1))).reshape(-1)                    
                    sources, targets = nn.nonzero()
                    r = np.array([total_link[x] for x in sources])
                    v = nn.data.flatten() * r
                    scaled = sparse.csr_matrix((v, (sources,targets)), shape=nn.shape, dtype='float')

                    logging.info(f'Pooling')
                    dsi["pooled"] = 'int32'
                    progress = tqdm(total = dsi.shape[0])
                    if "spliced" in dsi.layers:
                        dsi["spliced_pooled"] = 'int32'
                        dsi["unspliced_pooled"] = 'int32'
                        for (_, indexes, view) in dsi.scan(axis=0, layers=["spliced", "unspliced"], what=["layers"]):
                            dsi["spliced_pooled"][indexes.min(): indexes.max() + 1, :] = view.layers["spliced"][:, :] @ scaled.T
                            dsi["unspliced_pooled"][indexes.min(): indexes.max() + 1, :] = view.layers["unspliced"][:, :] @ scaled.T
                            dsi["pooled"][indexes.min(): indexes.max() + 1, :] = dsi["spliced_pooled"][indexes.min(): indexes.max() + 1, :] + dsi["unspliced_pooled"][indexes.min(): indexes.max() + 1, :]
                            progress.update(512)
                    else:
                        for (_, indexes, view) in dsi.scan(axis=0, layers=[""], what=["layers"]):
                            dsi["pooled"][indexes.min(): indexes.max() + 1, :] = view[:, :] @ scaled.T
                    progress.close()

                    logging.info(f'Set pooled as main layer')
                    dsi['raw'] = 'int32'
                    progress = tqdm(total = dsi.shape[0])
                    for (_, indexes, view) in dsi.scan(axis=1, what=["layers"]):
                        dsi['raw'][:,indexes.min(): indexes.max() + 1] = view[:,:]
                        dsi[''][:,indexes.min(): indexes.max() + 1] = view['pooled'][:,:]
                        progress.update(512)
                    progress.close()

                    dsi.ca['NGenes'] = dsi.map([np.count_nonzero], axis=1)[0]

                logging.info(f"Inferring cell cycle")
                species = Species.detect(dsi)
                CellCycleAnnotator(species).annotate(dsi, layer='')
                cgplot.cell_cycle(dsi, os.path.join(self.outdir, self.name + "_cellcycle.png"))

    def annotate(self, min_cells:int=10, agg_spec=None, imputed:bool=False, layer=''):
        '''
        '''
        if agg_spec == None:
            agg_spec = {
            "Age": "mean",
            "Clusters": "first",
            "Class": "mode",
            "Total": "mean",
            "Sex": "tally",
            "Tissue": "tally",
            "Chemistry": "tally",
            "SampleID": "tally",
            "TissuePool": "first",
            "Outliers": "mean",
            "PCW": "mean"
            }

        if imputed == True:
            file = self.Imputed_file
        else:
            file = self.RNA_file

        if not os.path.isfile(file):
            logging.info(f'Generate {file} first!')
            return
            
        with loompy.connect(file) as ds:
            cells = ds.col_attrs["Clusters"] >= 0
            labels = ds.col_attrs["Clusters"][cells]
            n_labels = len(set(labels))

            logging.info(f'Aggregating file')
            ds.aggregate(self.RNA_agg, None, "Clusters", "mean", agg_spec, layer=layer)

            with loompy.connect(self.RNA_agg) as dsout:
                dsout.ca.NCells = np.bincount(labels, minlength=n_labels)[dsout.ca.Clusters]
                dsout.ca.Clusters_peaks = dsout.ca.Clusters
                dsout.ca.Clusters = np.arange(n_labels)
                d = {k:v for k, v in zip(dsout.ca.Clusters_peaks, dsout.ca.Clusters)}
                ds.ca.Clusters_peaks = ds.ca.Clusters
                ds.ca.Clusters = [d[x] for x in ds.ca.Clusters]

                logging.info("Computing nonzero fractions")
                enr = Enrichment()
                # dsout.layers["enrichment"] = enr.fit(dsout, ds)
                dsout.layers["nonzeros"] = enr.cluster_nonzeros(ds)
            
                logging.info("Computing cluster gene enrichment scores")
                fe = FeatureSelectionByMultilevelEnrichment(mask=Species.detect(ds).mask(dsout, ("cellcycle", "sex", "ieg", "mt")), layer=layer)
                ds.ra.Selected = fe.fit(ds)
                markers = ds.ra.Selected == 1
                dsout.layers["enrichment"] = fe.enrichment

                # Reorder the genes, markers first, ordered by enrichment in clusters
                logging.info("Permuting rows")
                mask = np.zeros(ds.shape[0], dtype=bool)
                mask[markers] = True
                # fetch enrichment from the aggregated file, so we get it already permuted on the column axis
                gene_order = np.zeros(ds.shape[0], dtype='int')
                gene_order[mask] = np.argmax(dsout.layer["enrichment"][mask, :], axis=1)
                gene_order[~mask] = np.argmax(dsout.layer["enrichment"][~mask, :], axis=1) + dsout.shape[1]
                gene_order = np.argsort(gene_order)
                ds.permute(gene_order, axis=0)
                dsout.permute(gene_order, axis=0)
        
                logging.info(f'Trinarizing')
                trinaries = Trinarizer(0.2).fit(dsout)
                dsout['trinaries'] = trinaries

                logging.info(f'Annotating')
                AutoAnnotator(self.config.paths.autoannotation, ds=dsout).annotate(dsout)

                logging.info("Computing auto-auto-annotation")
                AutoAutoAnnotator(n_genes=6).annotate(dsout)        

                ## Restore clusterlabels
                logging.info(f'Restoring cluster labels')
                for dsx in [ds,dsout]:
                    dsx.ca.Clusters_renumbered = dsx.ca.Clusters
                    dsx.ca.Clusters = dsx.ca.Clusters_peaks
                
                ## Remove undersampled clusters
                remove = dsout.ca.NCells < min_cells
                logging.info(f'Clusters with too few measurements: {dsout.ca.Clusters[remove]}')
                for k in ['AutoAnnotation', 'MarkerGenes', 'MarkerRobustness', 'MarkerSelectivity', 'MarkerSpecificity']:
                    new_attr = dsout.ca[k]
                    new_attr[remove] = ''
                    dsout.ca[k] = new_attr

        ## Export the labels to the peak_aggregate file
        logging.info(f'Tranferring labels to peak aggregate file')
        with loompy.connect(self.RNA_agg) as dsagg:
            with loompy.connect(self.peak_agg) as dsout:
                annot = np.repeat('', dsout.shape[1]).astype('U128')
                annot[dsagg.ca.Clusters] = dsagg.ca.AutoAnnotation
                dsout.ca.AutoAnnotation = annot

                annot = np.repeat('', dsout.shape[1]).astype('U128')
                annot[dsagg.ca.Clusters] = dsagg.ca.MarkerGenes
                dsout.ca.MarkerGenes = annot     

                ## Transfer the linkage to RNA file
                dsagg.attrs.linkage = dsout.attrs.linkage 
                
        with loompy.connect(self.peak_file) as ds:
            with loompy.connect(self.peak_agg) as dsagg:
                cgplot.manifold(ds, os.path.join(self.outdir, f"{self.name}_Annotations.png"), list(dsagg.ca.MarkerGenes), list(dsagg.ca.AutoAnnotation), embedding = self.config.params.main_emb)

    def generate_plots(self):
        '''
        '''
        logging.info(f'Generating plots for RNA layer')
        if os.path.isfile(self.Imputed_file):
            f1 = self.Imputed_file
        else:
            f1 = self.RNA_file

        with loompy.connect(f1) as ds:
            with loompy.connect(self.RNA_agg) as dsout:
                if ds.ca.Clusters.max() <= 500:
                    if "pooled" in ds.layers:
                        cgplot.TF_heatmap(ds, dsout, os.path.join(self.outdir, f"{self.name}_RNA_TFs_heatmap_pooled.pdf"), layer="pooled")
                        cgplot.markerheatmap(ds, dsout, os.path.join(self.outdir, f"{self.name}_RNA_markers_heatmap_pooled.pdf"), layer="pooled")
                    else:
                        cgplot.TF_heatmap(ds, dsout, os.path.join(self.outdir, f"{self.name}_RNA_TFs_heatmap.pdf"), layer="")
                        cgplot.markerheatmap(ds, dsout, os.path.join(self.outdir, f"{self.name}_RNA_markers_heatmap.pdf"), layer="")
                
                if not 'CellCycle' in ds.ca:
                    layer = 'pooled' if 'pooled' in ds.ca else ''
                    species = Species.detect(ds)
                    CellCycleAnnotator(species).annotate(ds, layer=layer)
                cgplot.cell_cycle(ds, os.path.join(self.outdir, f"{self.name}_cellcycle.png"))
                cgplot.attrs_on_TSNE(ds = ds,
									out_file=os.path.join(self.outdir, f"{self.name}_RNA_QC.png"), 
									attrs=["DoubletFinderFlag", "DoubletFinderScore", "TotalUMI", "NGenes", "unspliced_ratio", "MT_ratio"], 
									plot_title=["Doublet Flag", "Doublet Score", "UMI counts", "Number of genes", "Unspliced / Total UMI", "Mitochondrial / Total UMI"])

                if not 'subregions' in ds.ca:
                    ds.ca.subregions = ds.ca.regions
                ov = PlotOverview(out_file=os.path.join(self.outdir, f"{self.name}_overview.png"))
                ov.fit(ds, dsout, save=True)



