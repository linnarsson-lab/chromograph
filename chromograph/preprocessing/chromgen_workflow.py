import sys
import numpy as np
import json
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
ref = sys.argv[2]
logging.info(f'Looking for {sample} Cellranger outputs at {config.paths.cell_ranger}')
dirs = glob.glob(f'{config.paths.cell_ranger}/{sample}*')

if len(dirs) > 0:
    dirs = np.array([d for d in dirs if os.path.isdir(d)])
    dirs = dirs[[len(d.split('_')) == 4 for d in dirs]]
    n_flowcells = [len(d.split('_')[-2]) for d in dirs]
    dirs = np.array(dirs)[np.where(n_flowcells==np.max(n_flowcells))[0]]
    ID = [int(d.split('_')[-1]) for d in dirs]
    
    ## Exclude folders with the wrong reference
    for i, d in enumerate(dirs):
        try:
            f = os.path.join(d, 'outs/summary.csv')
            summary = np.genfromtxt(f, dtype=str, delimiter=',')
            summary = {str(k): str(v) for k, v in zip(summary[0,:], summary[1,:])}
            summary['reference_assembly'] = summary['Genome']#     indir = dirs[np.where(ID==np.max(ID))[0]][0]
        except:
            f = os.path.join(d, 'outs/summary.json')
            summary = json.load(open(f, 'r'))
            summary = {k: str(v) for k,v in summary.items()}
        if summary['reference_assembly'] != ref:
            ID[i] = 0

    indir = dirs[np.where(ID==np.max(ID))[0]][0]
    outdir = f"{config.paths.samples}/{sample}"
    logging.info(f'Using cellranger output: {indir}')
    logging.info(f'Saving to {outdir}')

    chromgen = Chromgen()
    chromgen.fit(indir = indir, bsize = bsize, outdir = outdir)

else:
    logging.info(f'Could not find sample')