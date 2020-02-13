import loompy
import os
import sys
import numpy as np
from datetime import datetime
import logging
sys.path.append('/home/camiel/chromograph/')
from chromograph.pipeline.Bin_analysis import *
from chromograph.peak_calling.peak_caller import *
from chromograph.pipeline import config

config = config.load_config()

samples = ['232_1', '232_2']
tissue = 'Cerebellum'
bsize = '5kb'

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

## Check if directory exists
if not os.path.isdir(config.paths.build):
    os.mkdir(config.paths.build)

logging.info(f'The build folder is {config.paths.build}')

## Merge Bin files
outfile = os.path.join(config.paths.build, tissue + '.loom')
inputfiles = [os.path.join(config.paths.samples, '10X' + sample, '10X' + sample + f"_{bsize}.loom") for sample in samples]

logging.info(f'Input files {inputfiles}')

loompy.combine(inputfiles, outfile)
logging.info('Finished combining loom-files')

## Run primary Clustering and embedding
with loompy.connect(outfile) as ds:
    ds.attrs['tissue'] = tissue
    bin_analysis = bin_analysis()
    bin_analysis.fit(ds, outdir=config.paths.build)
    
## Merge GA files
GA_file = os.path.join(config.paths.build, tissue + '_GA.loom')
inputfiles = [os.path.join(config.paths.samples, '10X' + sample, f'10X{sample}_GA.loom') for sample in samples]
for x in inputfiles:
    with loompy.connect(x) as ds:
        logging.info(f"{x} has shape{ds.shape}")
loompy.combine(inputfiles, GAfile, key = 'Accession')

## Call peaks
with loompy.connect(outfile) as ds:
    peak_caller = Peak_caller()
    peak_caller.fit(ds)