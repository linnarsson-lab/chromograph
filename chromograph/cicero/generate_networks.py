## Imports
import loompy
import matplotlib.pyplot as plt
import logging
import numpy as np
from chromograph.cicero.cicero import *

class generate_Coaccessibilty_networks():
    def __init__(self, ds):
        self.peak_file = ds.filename
        self.name = ds.filename.split('/')[-2]
        self.out_dir = '/' + os.path.join(*ds.filename.split('/')[:-1], 'exported')
        self.pkl_file = '/' + os.path.join(*ds.filename.split('/')[:-1], 'exported', 'coaccess_matrix.pkl')

    def fit(self):
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        with loompy.connect(self.peak_file) as ds:
            if not 'pos' in ds.ra:
                ds.ra.pos = np.ceil((ds.ra.Start.astype(int) + ds.ra.End.astype(int))/2).astype(int)

            dist_param = estimate_distance_parameter(ds, verbose=False)
            plt.hist(dist_param)
            alpha = np.mean(dist_param)
            logging.info(f'Set alpha at {alpha}')
            matrix = Compute_Coacces(ds, alpha=alpha)
            df, filtered_matrix = generate_ccans(matrix, ds.ra.ID)
            pkl.dump(filtered_matrix, open(self.pkl_file, 'wb'))
            save_connections(ds, df, self.out_dir)