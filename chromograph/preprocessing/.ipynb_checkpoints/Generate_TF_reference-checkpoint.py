import numpy as np
import os
import sys
import collections
import matplotlib.pyplot as plt
import gzip
import pybedtools
from pybedtools import BedTool
import warnings

sys.path.append('/home/camiel/chromograph/')
import chromograph
from chromograph.peak_calling.utils import *

import cytograph as cg
from typing import *
from pybedtools.featurefuncs import *

def add_gene(f, gene):
    """
    adds name to feature
    """
    f.name = gene
    return f

def add_reliability(f, x):
    """
    Adds reliability as 'score' metric in bed.
    Reliability scores should correspond to:
    A: 3
    B: 2
    C: 1
    D: 0
    """
    f.score = x
    return f

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

## Set directory
cdir = '/data/proj/scATAC/chromograph/cistromes/cistrome_hg38'
motifdir = os.path.join(cdir, os.listdir(cdir)[2])

## Make list of files and unique TFs
files = os.listdir(motifdir)
TFs = np.unique([x.split('.')[0] for x in files])

## Only retain TF binding sites with at least 2 biological replicates or 2 technical replicates
valid = []
for TF in TFs:
    if f"{TF}.A.bed" in files:
        valid.append(TF)
    elif f"{TF}.B.bed" in files:
        valid.append(TF)
    elif f"{TF}.C.bed" in files:
        valid.append(TF)
        
discarded = [x for x in TFs if x not in valid]
logging.info(f"TFs kept: {len(valid)}   TFs discarded: {len(discarded)}")

## Encode certainty level on scale of 0-3
rel_dict = {'A': 3, 'B': 2, 'C': 1, 'D': 0}

rel = {}
i = 0
M = None

## Merge beds to reference of TF binding sites
for TF in TFs:
    for x in ['A','B','C']:
        if f"{TF}.{x}.bed" in files:
            cis = BedTool(os.path.join(motifdir, f"{TF}.{x}.bed")).saveas()
            cis = cis.each(extend_fields, 5).each(add_gene, TF.split('_')[0]).each(add_reliability, rel_dict[x]).saveas()
            if M != None:
                M = M.cat(*[cis], postmerge=False).saveas()
            else:
                M = cis.saveas()
            try:
                rel[TF] += 1
            except:
                rel[TF] = 1
    i += 1
    if i%10 == 0:
        logging.info(f"Finished {i} out of {len(valid)}")
logging.info(f"Total length: {len(M)} for {len(rel)} Motifs ")

## Save file as reference
M.saveas('/data/proj/scATAC/chromograph/cistromes/cistrome_hg38/cismotifs_ref.bed')