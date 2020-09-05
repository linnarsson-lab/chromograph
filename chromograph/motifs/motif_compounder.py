import os
import logging
import numpy as np
from typing import *
import sys

import loompy
from tqdm import tqdm

sys.path.append('/home/camiel/chromograph')
import chromograph
from chromograph.pipeline import config
from chromograph.peak_calling.utils import read_HOMER_TFs
from chromograph.pipeline.utils import div0
from chromograph.pipeline.TF_IDF import TF_IDF
from cytograph.manifold import BalancedKNN

class Motif_compounder:
    def __init__(self, outdir) -> None:
        """
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:
            ds                    Loom connection with peak file

        Remarks:
        
        """
        self.config = chromograph.pipeline.config.load_config()
        self.peakdir = os.path.join(outdir, 'peaks')
        self.outdir = outdir
        logging.info("Motif compounder initialised")

    def fit(self, ds: loompy.LoomConnection, agg_spec: Dict[str, str] = None) -> None:
        '''
        Generates loom-file with Motif enrichments based on HOMER-annotated peaks
        and peaks loomfile.
        
        Args:

            ds                      Loom connection with peak file
        Returns:
        
        Remarks:
        
        '''
        
        ## Get paths
        name = ds.filename.split('/')[-1].split(".")[0].split('_')[0]
        self.out_file = os.path.join(self.outdir, f"{name}_motifs.loom")
        self.agg_file = os.path.join(self.outdir, f"{name}_motifs.agg.loom")
        
        if 'NCells' not in ds.ra or 'NPeaks' not in ds.ca:
            logging.info('Calculating peak and cell coverage')
            ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
            ds.ca['NPeaks'] = ds.map([np.count_nonzero], axis=1)[0]

        ## Load the annotated peaks
        cols, table, TF_cols, TFs = read_HOMER_TFs(os.path.join(self.peakdir, 'motif_annotation.txt'))
        logging.info(f'Creating a loom-file to fill with enrichments of {len(TF_cols)} motifs for {ds.shape[1]} cells')
        
        ## Check All_peaks.loom exists, get subset
        all_motif = os.path.join(self.config.paths.build, 'All', 'All_motifs.loom')

        if os.path.exists(all_motif) & (all_motif != self.out_file):
            logging.info(f'Main motif matrix already exists')
            
            with loompy.connect(all_motif) as dsp:
                selection = np.array([x in ds.ca.CellID for x in dsp.ca.CellID])
            
            loompy.combine_faster([all_motif], self.out_file, selections=[selection])
            
            with loompy.connect(self.out_file) as ds2:
                transfer_ca(ds, ds2, 'CellID')
            logging.info(f'Finished creating promoter file')

        else:
            with loompy.new(self.out_file) as dsout:
                ## Transferring column attributes and grapsh from peak-file
                logging.info(f'Shape will be {TFs.shape[1]} rows by {ds.shape[1]} columns')
                dsout.add_columns(np.zeros([TFs.shape[1], ds.shape[1]]), col_attrs=ds.ca, row_attrs={'Gene': np.array([x.split('_')[0] for x in TF_cols]), 'Total_peaks': np.array(np.sum(TFs, axis = 0))})
                dsout.col_graphs = ds.col_graphs
                logging.info(f'New loom file has shape {dsout.shape}')

                ## Compound to motif enrichments
                logging.info(f'Compounding peaks to motif enrichments')
                progress = tqdm(total=ds.shape[1])
                for (ix, selection, view) in ds.scan(axis=1, batch_size=self.config.params.batch_size):
                    for x in range(len(TF_cols)):
                        dsout[x,selection] = np.sum(view[TFs[:,x], :], axis=0)
                    progress.update(self.config.params.batch_size)
                progress.close()

                logging.info('Normalizing against total peaks')
                dsout.layers['MMP'] = div0(dsout[:,:], (1e-6 * ds.ca['NPeaks']))

        with loompy.connect(self.out_file) as dsout:
            ## Smooth motif enrichments
            logging.info(f'Loading the network')
            bnn = BalancedKNN(k=self.config.params.k, metric='euclidean', maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
            bnn.bknn = dsout.col_graphs.KNN

            logging.info('Smoothing over the graph')
            dsout['smooth'] = 'float16'
            dsout['smooth'] = bnn.smooth_data(dsout['MMP'][:,:], only_increase=False)
            logging.info(f'Finished smoothing')

            ## Calculating modified Z-score
            logging.info('Generating modified Z-scores')
            dsout.ra['Median'] = dsout['smooth'].map([np.median], axis=0)[0]
            dsout.ra['MADS'] = np.median(abs(dsout['smooth'][:,:] - dsout.ra['Median'].reshape([dsout.shape[0],1])), axis=1)
            dsout['MZ'] = 0.6745 * div0(dsout['smooth'][:,:] - dsout.ra['Median'].reshape([dsout.shape[0],1]), dsout.ra['MADS'].reshape([dsout.shape[0],1]))
            dsout['counts'] = dsout[''][:,:]
            dsout[''] = dsout['MZ'][:,:]

            logging.info(f'Finished compounding motifs')

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
            cells = dsout.col_attrs["Clusters"] >= 0
            labels = dsout.col_attrs["Clusters"][cells]
            n_labels = len(set(labels))

            logging.info("Aggregating clusters")
            dsout.aggregate(self.agg_file, None, "Clusters", "mean", agg_spec)

        return self.out_file

