import numpy as np
import os
import sys
import collections

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *
import chromograph

def count_genes(data, rows, res):
    cell = data[0]
    frags = data[1]
    counts = {}

    for _frag in frags:
        for x in rows['Gene']:
            st = rows['Start'][rows['Gene'] == x]
            en = rows['End'][rows['Gene'] == x]
            if _frag[0] == rows['Chromosome'][rows['Gene'] == x]:
                if np.logical_and(_frag[1] > st, _frag[2] < en) or np.logical_and(_frag[1] < en, _frag[2] > en) or np.logical_and(_frag[1] < st, _frag[2] > st):
                    if x not in counts.keys():
                        counts[x] = 1
                    else:
                        counts[x] += 1

    res[cell] = counts