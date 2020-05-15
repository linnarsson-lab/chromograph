import os
import logging
import numpy as np
import sys

import loompy
from tqdm import tqdm

sys.path.append('/home/camiel/chromograph')
import chromograph
from chromograph.pipeline import config
from chromograph.peak_calling.utils import read_HOMER_annotation
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
        self.out_file = os.path.join(outdir, f"{outdir.split('/')[0]}_motifs.loom")
        logging.info("Motif compounder initialised")

    def fit(self, ds: loompy.LoomConnection) -> None:
        '''
        Generates loom-file with Motif enrichments based on HOMER-annotated peaks
        and peaks loomfile.
        
        Args:

            ds                      Loom connection with peak file
        Returns:
        
        Remarks:
        
        '''
        
        ## Get paths
        name = ds.filename.split(".")[0]
        
        if 'NCells' not in ds.ra or 'NPeaks' not in ds.ca:
            logging.info('Calculating peak and cell coverage')
            ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
            ds.ca['NPeaks'] = ds.map([np.count_nonzero], axis=1)[0]

        ## Load the annotated peaks
        cols, table, TF_cols, TFs = read_HOMER_annotation(os.path.join(self.peakdir, 'annotated_peaks.txt'))
        logging.info(f'Creating a loom-file to fill with enrichments of {len(TF_cols)} motifs for {ds.shape[1]} cells')
        
        with loompy.new(self.out_file) as dsout:
            ## Transferring column attributes and grapsh from peak-file
            logging.info(f'Shape will be {TFs.shape[1]} rows by {ds.shape[1]} columns')
            dsout.add_columns(np.zeros([TFs.shape[1], ds.shape[1]]), col_attrs=ds.ca, row_attrs={'Gene': np.array(TF_cols), 'Total_peaks': np.array(np.sum(TFs, axis = 0))})
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
        return self.out_file

