import os
import logging
import numpy as np

import loompy

sys.path.append('/home/camiel/chromograph')
import chromograph
from chromograph.pipeline import config
from chromograph.peak_calling.utils import read_HOMER
from chromograph.pipeline.utils import div0
from chromograph.pipeline.TF_IDF import TF_IDF
from cytograph.manifold import BalancedKNN

class motif_compounder:
    def __init__(self) -> None:
        """
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:
            ds                    Loom connection with peak file

        Remarks:
        
        """
        self.config = chromograph.pipeline.config.load_config()
        self.peakdir = os.path.join(self.config.paths.build, 'peaks')
        logging.info("Peak Caller initialised")

    def fit(self, ds: loompy.LoomConnection) -> None:
        '''
        Generates loom-file with Motif enrichments based on HOMER-annotated peaks
        and peaks loomfile.
        
        Args:

            ds                      Loom connection with peak file
        Returns:
        
        Remarks:
        
        '''
        
        ## Load the annotated peaks
        cols, table, TF_cols, TFs = read_HOMER(os.path.join(self.peakdir, 'annotated_peaks.txt'))

        logging.info(f'Creating a loom-file to fill with motif enrichments for cells')
        f_out = os.path.join(self.config.paths.build, ds.attrs['tissue'] + '_motifs.loom')
        with loompy.new(f_out) as dsout:

            ## Transferring column attributes and grapsh from peak-file
            dsout.add_columns(np.zeros([TFs.shape[1], ds.shape[1]]), col_attrs=ds.ca, row_attrs={'Gene': np.array(TF_cols), 'Total_peaks': np.array(np.sum(TFs, axis = 0))})
            dsout.col_graphs = ds.col_graphs
            logging.info(f'New loom file has shape{dsout.shape}')

            ## Compound to motif enrichments
            logging.info(f'Compounding peaks to motif enrichments')
            for (ix, selection, view) in ds.scan(axis=1):
                for x in range(len(TF_cols)):
                    dsout[x,selection] = np.sum(view[TFs[:,x], :], axis=0)
                logging.info(f'finished {max(selection)}')

            ## TF-IDF
            tf_idf = TF_IDF()
            tf_idf.fit(ds)
            dsout['TF_IDF'] = tf_idf.transform(ds[''][:,:])

            logging.info('Normalizing against depth')
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
        return f_out

