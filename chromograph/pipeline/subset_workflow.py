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
from chromograph.preprocessing.utils import get_blacklist, mergeBins
from chromograph.pipeline import config
from chromograph.pipeline.Bin_analysis import *
from chromograph.pipeline.utils import transfer_ca
from chromograph.pipeline.Add_UMAP import Add_UMAP
from chromograph.peak_calling.peak_caller import *
from chromograph.peak_calling.utils import *
from chromograph.peak_analysis.peak_analysis import Peak_analysis
from chromograph.peak_analysis.Peak_Aggregator import Peak_Aggregator
from chromograph.RNA.RNA_analysis import RNA_analysis
from chromograph.features.Generate_promoter import Generate_promoter
from chromograph.features.GA_Aggregator import GA_Aggregator
from chromograph.motifs.motif_compounder import Motif_compounder
from chromograph.motifs.motif_aggregation import motif_aggregator

## Import punchcards
from cytograph.pipeline.punchcards import (Punchcard, PunchcardDeck, PunchcardSubset, PunchcardView)

## Setup logger and load config
config = config.load_config()
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

current_set = sys.argv[1]

if __name__ == '__main__':

    logging.info(f'Starting subset workflow')
    bsize = f'{int(config.params.bin_size/1000)}kb'

    ## Load punchcard and setup decks for analysis
    deck = PunchcardDeck(config.paths.build)

    subset = deck.get_subset(current_set)
    name = subset.name
    samples = subset.include

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
    Skip_bins = False

    ## Check if bin analysis should be skipped
    if not name == 'All':
        main_peaks = os.path.join(os.path.join(config.paths.build, 'All', 'All_peaks.loom'))
        if os.path.isfile(main_peaks):
            Skip_bins = True
            logging.info(f'Skip binning')

    if ('bin_analysis' in config.steps) & (Skip_bins != True):
        ## Add UMAP to main loom
        if name == 'All':
            with loompy.connect(binfile, 'r+') as ds:
                new_UMAP = Add_UMAP(subset_dir, 'bins')
                new_UMAP.fit(ds)
        else:
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
                    good_cells = (ds.ca.DoubletFinderFlag == 0) & (ds.ca.passed_filters > 5000) & (ds.ca.passed_filters < 1e5) & (ds.ca.TSS_fragments/ds.ca.passed_filters > config.params.FR_TSS)
                    selections.append(good_cells)

            ## Merge Bin files
            if not os.path.exists(binfile):

                ## Get column attributes that should be skipped
                skip_attr = find_attr_to_skip(config, samples)
                skip_attr = set(config.params.skip_attrs + skip_attr)
                logging.info(f'Not including the following column attributes {skip_attr}')

                logging.info(f'Input samples {samples}')
                loompy.combine_faster(inputfiles, binfile, selections=selections, key = 'loc', skip_attrs=skip_attr)
                # loompy.combine(inputfiles, outfile, key = 'loc')       ## Use if running into memory errors
                logging.info('Finished combining loom-files')
            else:
                logging.info('Combined bin file already exists, using this for analysis')

            ## Run primary Clustering and embedding
            with loompy.connect(binfile, 'r+') as ds:
                bin_analysis = Bin_analysis(outdir=subset_dir, do_UMAP=config.params.UMAP)
                bin_analysis.fit(ds) 

    ## Check if peak subset should be taken from main file
    if (not name == 'All') & (not os.path.isfile(peak_file)):
        logging.info(f'Main peak matrix already exists, taking subset')
        with loompy.connect(main_peaks, 'r') as ds_main:
            selection = np.array([x.split('10X')[-1] in samples for x in ds_main.ca.Name])
            logging.info(f'Cells collected: {np.sum(selection)}')
        loompy.combine_faster([main_peaks], peak_file, selections=[selection])
        
        ## Set broad clustering as preclusters
        with loompy.connect(peak_file) as ds:
            logging.info(f'Set broad clustering as preclusters')
            ds.ca.preClusters = ds.ca.Clusters
            del ds.ca.Clusters
        logging.info(f'Finished creating peak file')

    ## Analyse peak-file
    if 'peak_analysis' in config.steps:

        ## Add UMAP to main loom
        if name == 'All':
            with loompy.connect(peak_file, 'r+') as ds:
                new_UMAP = Add_UMAP(subset_dir, 'peaks')
                new_UMAP.fit(ds)

            logging.info(f'Transferring column attributes and column graphs back to bin file')
            with loompy.connect(peak_file, 'r+') as ds:
                with loompy.connect(binfile) as dsb:
                    transfer_ca(ds, dsb, 'CellID')

        else:
            peak_agg = os.path.join(subset_dir, name + '_peaks.agg.loom')
            with loompy.connect(peak_file) as ds:
                peak_analysis = Peak_analysis(outdir=subset_dir, do_UMAP=config.params.UMAP)
                peak_analysis.fit(ds)

                peak_aggregator = Peak_Aggregator()
                peak_aggregator.fit(ds, peak_agg)

    if 'RNA' in config.steps:
        ## Generate RNA imputation file and annotation
        with loompy.connect(peak_file) as ds:

            if len(np.where(ds.ca.Chemistry=='multiome_atac')[0]) > 0:
                RNA_imputer = RNA_analysis(ds, outdir=subset_dir)
                RNA_imputer.generate_RNA_file(config.paths.RNA) ## Generate RNA file
                if 'Impute_RNA' in config.steps:
                    RNA_imputer.Impute_RNA() ## Impute RNA on non-RNA samples
                RNA_imputer.annotate() ## Aggregate and annotate clusters
            else:
                logging.info(f'No Multiome cells in subset, skipping step')

    if 'prom' in config.steps:
        ## Generate promoter file
        logging.info(f'Generating promoter file')
        with loompy.connect(peak_file, 'r') as ds:
            Promoter_generator = Generate_promoter(outdir=subset_dir, poisson_pooling=config.params.poisson_pooling)
            GA_file = Promoter_generator.fit(ds)

        ## Transer column attributes
        with loompy.connect(GA_file) as ds:
            ## Aggregate GA file and annotate based on markers
            GA_agg_file = os.path.join(subset_dir, name + '_prom.agg.loom')
            Aggregator = GA_Aggregator()
            Aggregator.fit(ds, out_file=GA_agg_file)

            logging.info(f'Transferring column attributes back to peak file')
            with loompy.connect(peak_file) as dsb:
                transfer_ca(ds, dsb, 'CellID')

    if 'motifs' in config.steps:
        with loompy.connect(peak_file) as ds:
            MA = motif_aggregator(name)
            MA.fit()

    if 'cicero' in config.steps:
        cicero_run = os.path.join('/', *chromograph.__file__.split('/')[:-1], 'cicero', 'run_cicero.py')
        subprocess.run([config.paths.cicero_path, cicero_run, peak_file, 'True'])

    ## Export bigwigs last to prevent multiprocessing error
    if 'bigwig' in config.steps:
        ## Export bigwigs by cluster
        with loompy.connect(peak_file, 'r') as ds:
            logging.info(f'Exporting bigwigs for {name}')
            with mp.get_context().Pool(20) as pool:
                for cluster in np.unique(ds.ca.Clusters):
                    cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
                    pool.apply_async(export_bigwig, args=(cells, config.paths.samples, os.path.join(subset_dir, 'peaks'), cluster,))
                pool.close()
                pool.join()
        logging.info(f'Finished saving bigwigs')