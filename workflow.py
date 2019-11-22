## Test chromograph Binning

import numpy as np
import os
import sys
import pybedtools
from pybedtools import BedTool
import collections
import csv
import matplotlib.pyplot as plt
import gzip
import loompy
import scipy.sparse as sparse
import json
import urllib.request

from chromograph.preprocessing.bin_generation import *
from chromograph.pipeline.Bin_analysis import *

import logging

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

bsize = 10000
indir = '/data/proj/scATAC/10X_test/atac_v1_E18_brain_fresh_5k_S1_cell_ranger'
outdir = '/data/proj/scATAC/chromograph/mouse_test2'

chrombin = Chrombin()
chrombin.fit(indir = indir, bsize = bsize, outdir = outdir)

with loompy.connect(chrombin.loom) as ds:
    bin_analysis = bin_analysis()
    bin_analysis.fit(ds, outdir=outdir)