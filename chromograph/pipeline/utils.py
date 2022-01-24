import numpy as np
import os
import loompy
import logging

def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c

def transfer_ca(ds1: loompy.LoomConnection, ds2: loompy.LoomConnection, key: str = 'CellID'):
    '''
    Transfers the column attributes from ds1 to ds2. Both loom-files must
    have the same number of columns with a corresponding unique identifies
    Args:
        ds1         LoomConnection
        ds2         LoomConnection, to be reordered
        key         Column attriube used to align the columns
    '''

    if np.array_equal(ds1.ca.CellID, ds2.ca.CellID):
        logging.info('Datasets already ordered')
    else:
        ## Align the datasets
        logging.info('Permuting dataset 2 based on order of dataset 1')
        match = {k:v for v, k in enumerate(ds1.ca[key])}
        new_index = np.array([match[x] for x in ds2.ca[key]]).argsort()
        ds2.permute(new_index, axis=1)
        logging.info(f'Finished permuting datasets')

    ## Transfer column attributes
    for x in ds1.ca:    ## Don't check if attribute already present as it will prevent transfer of TSNE coordinates
        ds2.ca[x] = ds1.ca[x]

    ## Transfer column graphs
    for x in ds1.col_graphs:
        ds2.col_graphs[x] = ds1.col_graphs[x]
    return

def find_attr_to_skip(config, samples):
    '''
    Cycle throught all samples given as input and return column attributes that
    don't appear in all samples or have incosistent dtypes
    '''
    vars = {}
    vars_dtype = {}

    for i, sample in enumerate(samples):
        f = os.path.join(config.paths.samples, f'10X{sample}', f'10X{sample}_5kb.loom')

        with loompy.connect(f, 'r') as ds:
            ## In first round add all the column attributes
            if i == 0:
                for k in ds.ca:
                    if k not in vars:
                        vars[k] = True
                        vars_dtype[k] = ds.ca[k].dtype

            ## In other rounds check for all saved attributes if they exist in this file and have the same dtype
            else:
                for k in ds.ca:
                    if k not in vars:
                        vars[k] = False
                        vars_dtype[k] = ds.ca[k].dtype
                for k in vars:
                    if k not in ds.ca:
                        vars[k] = False
                    elif ds.ca[k].dtype != vars_dtype[k]:
                        vars[k] = False

    return [k for k, v in vars.items() if v == False]