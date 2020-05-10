import numpy as np
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
        ds2         LoomConnection
        key         Column attriube used to align the columns
    '''

    if np.array_equal(ds1.ca.CellID, ds2.ca.CellID):
        logging.info('Datasets already ordered')
    else:
        ## Align the datasets
        logging.info('Permuting dataset 2 based on order of dataset 1')
        match = {k:v for v, k in enumerate(ds.ca[key])}
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