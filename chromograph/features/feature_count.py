import numpy as np
import os
import sys
import collections

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *
import chromograph

def feature_count(ds, coding, res):
    i = 0
    logging.info(f"Started")
    
    fragments = ds.ca['fragments']
    logging.info(f"loaded fragments")
    fragments = [strFrags_to_list(x) for x in fragments]
    
    logging.info(f"extracted fragments")
    
    for cell, frags in zip(ds.ca['cell_id'], fragments):
        _frags = BedTool(frags)
        counts = coding.intersect(_frag, c=True)
        counts = [x[10] for x in counts]
        res[cell] = counts
        i += 1

            
        if i%300 == 0:
            logging.info(f"Counted {i} cells")