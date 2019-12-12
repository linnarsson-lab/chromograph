import loompy
import os
import sys
import numpy as np
from datetime import datetime
import logging
sys.path.append('/home/camiel/chromograph/')
from chromograph.pipeline.Bin_analysis import *

path = '/data/proj/scATAC/chromograph/'

samples = ['232_1', '232_2']
tissue = 'Cerebellum'
bsize = '5kb'
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
inputfiles = [os.path.join(path, '10X' + sample, '10X' + sample + f"_{bsize}.loom") for sample in samples]

for x in inputfiles:
    with loompy.connect(x) as ds:
        logging.info(f"{x} has shape{ds.shape}")
loompy.combine(inputfiles, outfile)

with loompy.connect(outfile) as ds:
    ds.attrs['tissue'] = tissue
    bin_analysis = bin_analysis()
    bin_analysis.fit(ds, outdir=outdir)