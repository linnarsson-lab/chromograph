## Imports
import loompy
import matplotlib.pyplot as plt
import logging
import numpy as np
from chromograph.cicero.cicero import *

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class generate_Coaccessibilty_networks():
    def __init__(self, ds, dsagg, generate_GA=False):
        self.peak_file = ds.filename
        self.peak_agg = dsagg.filename
        self.name = ds.filename.split('/')[-2]
        self.out_dir = '/' + os.path.join(*ds.filename.split('/')[:-1], 'exported')
        self.cicero_dir = '/' + os.path.join(*ds.filename.split('/')[:-1], 'exported')
        self.pkl_file = os.path.join(self.out_dir, 'coaccess_matrix.pkl')
        self.generate_GA = generate_GA

    def fit(self):
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        if not os.path.isdir(self.cicero_dir):
            os.mkdir(self.cicero_dir)

        with loompy.connect(self.peak_agg) as dsagg:
            if not 'pos' in dsagg.ra:
                dsagg.ra.pos = np.ceil((dsagg.ra.Start.astype(int) + dsagg.ra.End.astype(int))/2).astype(int)

            ## Estimate distance parameter
            dist_param = estimate_distance_parameter(dsagg, verbose=False)
            plt.hist(dist_param)
            alpha = np.mean(dist_param)
            logging.info(f'Set alpha at {alpha}')

            ## Calculate coaccessibility map
            matrix = Compute_Coacces(dsagg, alpha=alpha)

            ## Find local regions of coaccessibility
            df, filtered_matrix = generate_ccans(dsagg, matrix, dsagg.ra.ID)
            pkl.dump(filtered_matrix, open(self.pkl_file, 'wb'))
            save_connections(dsagg, df, self.out_dir)

            ## Generate the Gene Activity matrix
            if self.generate_GA:
                with loompy.connect(self.peak_file) as ds:
                    generate_Gene_Activity(ds, filtered_matrix)