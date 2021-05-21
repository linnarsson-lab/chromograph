import sys
import numpy as np
import os
import csv
import gzip
import loompy
import glob

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
logging.info(f'Looking for {sample} Cellranger outputs at {config.paths.cell_ranger}')
dirs = glob.glob(f'{config.paths.cell_ranger}/{sample}*')

if len(dirs) > 0:
    dirs = np.array([d for d in dirs if os.path.isdir(d)])
    dirs = dirs[[len(d.split('_')) == 4 for d in dirs]]
    n_flowcells = [len(d.split('_')[-2]) for d in dirs]
    dirs = np.array(dirs)[np.where(n_flowcells==np.max(n_flowcells))[0]]
    ID = [int(d.split('_')[-1]) for d in dirs]
    
    indir = dirs[np.where(ID==np.max(ID))[0]][0]
    outdir = f"{config.paths.samples}/{sample}"
    logging.info(f'Using cellranger output: {indir}')
    logging.info(f'Saving to {outdir}')

    chromgen = Chromgen()
    chromgen.fit(indir = indir, bsize = bsize, outdir = outdir)

else:
    logging.info(f'Could not find sample')