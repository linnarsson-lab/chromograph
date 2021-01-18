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
if os.path.exists(f"/data/proj/chromium/{sample}_AB_1"):
    indir = f"/data/proj/chromium/{sample}_AB_1"
    logging.info('Using AB file')
elif os.path.exists(f"/data/proj/chromium/{sample}_A_1"):
    indir = f"/data/proj/chromium/{sample}_A_1"
    logging.info('Using A file')
elif os.path.exists(f"/data/proj/chromium/{sample}"):
    indir = f"/data/proj/chromium/{sample}"
    logging.info('Using	unlabelled file')
else:
    logging.info("Could not find sample")
# outdir = f"/data/proj/scATAC/samples/{sample}"
outdir = os.path.join(config.paths.samples, sample)

chromgen = Chromgen()
chromgen.fit(indir = indir, bsize = bsize, outdir = outdir)