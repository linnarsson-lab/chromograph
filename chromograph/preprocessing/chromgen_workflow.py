import sys
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

from chromograph.pipeline import config
from chromograph.preprocessing.Chromgen import *
import logging

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

bsize = 5000
config = config.load_config()

sample = sys.argv[1]
CR_outputs = config.paths.cell_ranger
logging.info(f'Looking for Cellranger outputs at {CR_outputs}')
if os.path.exists(os.path.join(CR_outputs, f"{sample}_AB_1")):
    indir = os.path.join(CR_outputs, f"{sample}_AB_1")
    logging.info('Using AB file')
elif os.path.exists(os.path.join(CR_outputs, f"{sample}_A_1")):
    indir = os.path.join(CR_outputs, f"{sample}_A_1")
    logging.info('Using A file')
elif os.path.exists(os.path.join(CR_outputs, f"{sample}")):
    indir = os.path.join(CR_outputs, f"{sample}")
    logging.info('Using	unlabelled file')
else:
    logging.info("Could not find sample")
# outdir = f"/data/proj/scATAC/samples/{sample}"
outdir = os.path.join(config.paths.samples, sample)

chromgen = Chromgen()
chromgen.fit(indir = indir, bsize = bsize, outdir = outdir)