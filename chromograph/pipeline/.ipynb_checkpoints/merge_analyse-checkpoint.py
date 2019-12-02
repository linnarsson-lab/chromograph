import loompy
import os
import sys
import numpy as np
from datetime import datetime
import logging
sys.path.append('/home/camiel/chromograph/')
from chromograph.pipeline.Bin_analysis import *

path = '/data/proj/scATAC/chromograph/'

samples = ['232_3', '232_4']
tissue = 'Midbrain'
d = datetime.today().strftime('%Y%m%d')
outdir = os.path.join(path, 'build_' + d)

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

if not os.path.isdir(outdir):
    os.mkdir(outdir)

outfile = os.path.join(outdir, tissue + '.loom')
inputfiles = [os.path.join(path, '10X' + sample, '10X' + sample + '_10kb.loom') for sample in samples]
loompy.combine(inputfiles, outfile)

with loompy.connect(outfile) as ds:
    bin_analysis = bin_analysis()
    bin_analysis.fit(ds, outdir=outdir)