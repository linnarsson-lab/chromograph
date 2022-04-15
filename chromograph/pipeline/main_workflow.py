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
from chromograph.pipeline.Bin_analysis import *
from chromograph.pipeline import config
from chromograph.peak_analysis.peak_analysis import Peak_analysis
from chromograph.pipeline.utils import *
from chromograph.preprocessing.utils import get_blacklist, mergeBins
from chromograph.features.Generate_promoter import Generate_promoter
from chromograph.features.gene_smooth import GeneSmooth
from chromograph.features.GA_Aggregator import GA_Aggregator
from chromograph.peak_calling.peak_caller import *
from chromograph.peak_calling.utils import *
from chromograph.peak_calling.call_MACS import call_MACS
from chromograph.plotting.peak_annotation_plot import *
from chromograph.motifs.motif_compounder import Motif_compounder
from chromograph.peak_analysis.Peak_Aggregator import Peak_Aggregator
from chromograph.CNV.Karyotyper import Karyotyper

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

    logging.info(f'Starting main workflow')
    bsize = f'{int(config.params.bin_size/1000)}kb'
    ## Load punchcard and setup decks for analysis
    deck = PunchcardDeck(config.paths.build)
    for subset in deck.get_leaves():
        name = subset.longname()
        samples = subset.include

        if not name == 'All':
            continue

        logging.info(f'Performing the following steps: {config.steps} for build {config.paths.build}')
        ## Check if directory exists
        if not os.path.isdir(config.paths.build):
            os.mkdir(config.paths.build)
        
        logging.info(f'Starting analysis for subset {name}')
        subset_dir = os.path.join(config.paths.build, name)
        if not os.path.isdir(subset_dir):
            os.mkdir(subset_dir)
        binfile = os.path.join(subset_dir, name + '.loom')
        peak_file = os.path.join(subset_dir, name + '_peaks.loom')
        peak_agg = os.path.join(subset_dir, name + '_peaks.agg.loom')

        if 'bin_analysis' in config.steps:
            ## Select valid cells from input files
            inputfiles = [os.path.join(config.paths.samples, '10X' + sample, '10X' + sample + f"_{bsize}.loom") for sample in samples]
            selections = []
            for file in inputfiles:

                ## Check if file with right binning exists
                if not os.path.exists(file):
                    file_5kb = os.path.join(os.path.dirname(file), f'{file.split("/")[-2]}_5kb.loom')
                    mergeBins(file_5kb, config.params.bin_size)

                ## Get cells passing filters
                with loompy.connect(file, 'r') as ds:
                    if ds.ca.Chemistry[0] == 'multiome_atac':
                        good_cells = (ds.ca.DoubletFinderFlag == 0) & (ds.ca.passed_filters > config.params.level) & (ds.ca.passed_filters < config.params.max_fragments) & (ds.ca.TSS_fragments/ds.ca.passed_filters > config.params.FR_TSS) & (ds.ca.TotalUMI > config.params.min_umis) & (ds.ca.MT_ratio < config.params.max_fraction_MT_genes) & (ds.ca.unspliced_ratio > config.params.min_fraction_unspliced_reads)
                    else:
                        good_cells = (ds.ca.DoubletFinderFlag == 0) & (ds.ca.passed_filters > config.params.level) & (ds.ca.passed_filters < config.params.max_fragments) & (ds.ca.TSS_fragments/ds.ca.passed_filters > config.params.FR_TSS)
                    selections.append(good_cells)

            ## Get column attributes that should be skipped
            skip_attr = find_attr_to_skip(config, samples)
            skip_attr = set(config.params.skip_attrs + skip_attr)
            logging.info(f'Not including the following column attributes {skip_attr}')

            # ## Merge Bin files
            if not os.path.exists(binfile):
                logging.info(f'Input samples {samples}')
                loompy.combine_faster(inputfiles, binfile, selections=selections, key = 'loc', skip_attrs=skip_attr)
                # loompy.combine(inputfiles, outfile, key = 'loc')       ## Use if running into memory errors
                logging.info('Finished combining loom-files')
            else:
                logging.info('Combined bin file already exists, using this for analysis')

            ## Run primary Clustering and embedding
            with loompy.connect(binfile, 'r+') as ds:
                bin_analysis = Bin_analysis(outdir=subset_dir, do_UMAP=False)
                bin_analysis.fit(ds)

        if 'peak_calling' in config.steps:
            ## Call peaks
            with loompy.connect(binfile, 'r+') as ds:
                peak_caller = Peak_caller(outdir=subset_dir)
                peak_caller.fit(ds)

        if 'peak_analysis' in config.steps:
            ## Analyse peak-file
            with loompy.connect(peak_file, 'r+') as ds:
                # peak_analysis = Peak_analysis(outdir=subset_dir, do_UMAP=False)
                peak_analysis = Peak_analysis(outdir=subset_dir, do_UMAP=True)
                peak_analysis.fit(ds)

                peak_aggregator = Peak_Aggregator()
                peak_aggregator.fit(ds, peak_agg)

                logging.info(f'Transferring column attributes and column graphs back to bin file')
                with loompy.connect(binfile) as dsb:
                    transfer_ca(ds, dsb, 'CellID')
        
        if 'Karyotype' in config.steps:
            with loompy.connect(peak_file) as ds:
                with loompy.connect(peak_agg, 'r') as dsagg:
                    Karyotyper = Karyotyper()
                    Karyotyper.fit(ds, dsagg)
                    Karyotyper.plot(ds, dsagg)

        if 'prom' in config.steps:
            ## Generate promoter file
            logging.info(f'Generating promoter file')
            with loompy.connect(binfile, 'r') as ds:
                Promoter_generator = Generate_promoter(outdir=subset_dir, poisson_pooling=False)
                GA_file = Promoter_generator.fit(ds)

    logging.info('Done with steps')