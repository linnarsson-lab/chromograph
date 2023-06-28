import loompy
import os
import subprocess
import gc
import sys
import numpy as np
from datetime import datetime
import logging
from typing import *
from tqdm import tqdm

import gzip
import glob
import pybedtools
from pybedtools import BedTool
import MACS2
import shutil
import multiprocessing as mp

## Import chromograph
import chromograph
from chromograph.pipeline import config
from chromograph.pipeline.Pool_split import Pool_Splits
from chromograph.pipeline.utils import transfer_ca

## Import punchcards
from cytograph.pipeline.punchcards import (Punchcard, PunchcardDeck, PunchcardSubset, PunchcardView)

## Setup logger and load config
config = config.load_config()
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

if __name__ == '__main__':

    f_pool = f'{config.paths.build}/Pool/Pool_peaks.loom'
    PW = Pool_Splits(deck=PunchcardDeck(config.paths.build), config=config)
    # PW.collect_cells()
    # PW.Pool_RNA()
    # PW.Aggregate_motifs()
    PW.export_bigwigs()

    logging.info(f'Finished steps')