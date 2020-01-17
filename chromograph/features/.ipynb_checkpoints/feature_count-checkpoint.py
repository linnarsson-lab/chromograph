import numpy as np
import os
import sys
import collections
import logging
import pybedtools

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *
import chromograph

def Count_genes(bars, intersections):
    '''
    '''
    Count_dict = {k: {} for k in bars}
    i = 0

    for x in intersections:

        bar = str(x[-2])
        g = x.attrs['gene_id']

        if bar in Count_dict:
            if g not in Count_dict[bar]:
                Count_dict[bar][g] = 1
            else:
                Count_dict[bar][g] += 1
        i += 1

        if i%1000000 == 0:
            logging.info(f'Counted {i/1000000} million Fragments')
            
    return Count_dict