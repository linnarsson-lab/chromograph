import numpy as np
import loompy

def div0(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c

def transfer_ca(ds1: loompy.LoomConnection, ds2: loompy.LoomConnection, key: str):
    '''
    Transfers the column attributes from ds1 to ds2. Both loom-files must
    have the same number of columns with a corresponding unique identifies
    Args:
        ds1         LoomConnection
        ds2         LoomConnection
        key         Column attriube used to align the columns
    '''
    ## Align the datasets
    ds1.permute(ds1.ca[key].argsort(), axis=1)
    ds2.permute(ds2.ca[key].argsort(), axis=1)
    
    ## Transfer column attributes
    for x in ds1.ca:
        if x not in ds2.ca:
            ds2.ca[x] = ds1.ca[x]

    ## Transfer column graphs
    for x in ds1.col_graphs:
        if x not in ds2.col_graphs:
            ds2.col_graphs[x] = ds1.col_graphs[x]
    return