import numpy as np
import os
import sys
import collections

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *
import chromograph

def feature_count(data, coding, res):
    logging.info(f"Started")
    
    _frags = BedTool(data[1])
    counts = coding.intersect(_frag, c=True)
    counts = [x[10] for x in counts]
    res[data[0]] = counts
    return