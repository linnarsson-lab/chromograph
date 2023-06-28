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
from chromograph.pipeline.Split_subset import split_subset
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

current_set = sys.argv[1]

if __name__ == '__main__':

    logging.info(f'Starting subset workflow')
    ## Load punchcard and setup decks for analysis
    deck = PunchcardDeck(config.paths.build)

    subset = deck.get_subset(current_set)
    name = subset.longname()

    ## Decide how to select data
    if subset.onlyif:
        select_by, operator, condition = subset.onlyif.split(' ')
        parent = '_'.join(subset.longname().split('_')[:-1])
        main_peaks = os.path.join(os.path.join(config.paths.build, parent, f'{parent}_peaks.loom'))
    elif len(subset.include) > 0:
        if subset.include[0][0].isdigit():
            select_by = 'Samples'
            samples = subset.include
            main_peaks = os.path.join(os.path.join(config.paths.build, 'All', 'All_peaks.loom'))
        else:
            select_by = 'Annotation'
            main_peaks = os.path.join(os.path.join(config.paths.build, 'All', 'All_peaks.loom'))
    else:
        select_by = 'Annotation'
        main_peaks = os.path.join(os.path.join(config.paths.build, 'All', 'All_peaks.loom'))

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

    ## Check if peak subset should be taken from main file
    if (not name == 'All') & (not os.path.isfile(peak_file)):
        logging.info(f'Main peak matrix already exists, taking subset')
        with loompy.connect(main_peaks, 'r') as ds_main:
            if select_by == 'Split':
                select = int(deck.get_subset(current_set).onlyif[-1])
                selection = np.array([x == select for x in ds_main.ca.Split])      
            elif select_by == 'Samples':
                selection = np.array([x.split('10X')[-1] in samples for x in ds_main.ca.Name])
            else:
                select = np.array(condition.split(',')).astype(ds_main.ca[select_by].dtype)
                if operator == '!=':
                    selection = np.array([x != select for x in ds_main.ca[select_by]]).flatten()
                elif operator == '==':
                    selection = np.array([x == select for x in ds_main.ca[select_by]]).flatten()
                elif operator == 'in':
                    selection = np.isin(ds_main.ca[select_by], select)
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

        with loompy.connect(peak_file) as ds:
            peak_analysis = Peak_analysis(outdir=subset_dir, do_UMAP=config.params.UMAP)
            peak_analysis.fit(ds)

            peak_aggregator = Peak_Aggregator()
            if name == 'All':
                peak_aggregator.fit(ds, peak_agg, motifs=True)
            else:
                peak_aggregator.fit(ds, peak_agg, motifs=False)

    if 'Karyotype' in config.steps:
        with loompy.connect(peak_file) as ds:
            with loompy.connect(peak_agg, 'r') as dsagg:
                Karyotyper = Karyotyper()
                Karyotyper.fit(ds, dsagg)
                Karyotyper.plot(ds, dsagg)
                Karyotyper.generate_punchcards(config, ds, dsagg, python_exe=config.paths.pythonexe)


    if 'RNA' in config.steps:
        ## Generate RNA imputation file and annotation
        with loompy.connect(peak_file) as ds:

            if len(np.where(ds.ca.Chemistry=='multiome_atac')[0]) > 100:
                RNA_imputer = RNA_analysis(ds, outdir=subset_dir)
                RNA_imputer.generate_RNA_file(config.paths.RNA) ## Generate RNA file
                imputed = True if ('Impute_RNA' in config.steps) & (len(np.where(ds.ca.Chemistry=='multiome_atac')[0]) < ds.shape[1]) else False
                if 'Impute_RNA' in config.steps:
                    RNA_imputer.Impute_RNA() ## Impute RNA on non-RNA samples
                RNA_imputer.annotate(imputed=imputed) ## Aggregate and annotate clusters
                RNA_imputer.generate_plots()
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
        logging.info(f'Start motif aggragation and plotting')
        with loompy.connect(peak_file) as ds:
            MA = motif_aggregator(name)
            MA.fit()

    if 'cicero' in config.steps:
        cicero_run = os.path.join('/', *chromograph.__file__.split('/')[:-1], 'cicero', 'run_cicero.py')
        subprocess.run([config.paths.cicero_path, cicero_run, peak_file, 'True'])

    ## Export bigwigs last to prevent multiprocessing error
    if 'bigwigs' in config.steps:
        bigwig_dir = os.path.join(subset_dir, 'bigwigs')
        if not os.path.isdir(bigwig_dir):
            os.mkdir(bigwig_dir)

        ## Export bigwigs by cluster
        with loompy.connect(peak_file, 'r') as ds:
            logging.info(f'Exporting bigwigs for {name}')
            with mp.get_context().Pool(min(mp.cpu_count(),5)) as pool:
                for cluster in np.unique(ds.ca.Clusters):
                    cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
                    pool.apply_async(export_bigwig, args=(cells, config.paths.samples, bigwig_dir, cluster,))
                pool.close()
                pool.join()
            # for cluster in np.unique(ds.ca.Clusters):
            #     cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
            #     export_bigwig(cells, config.paths.samples, bigwig_dir, cluster)

        with loompy.connect(self.peak_agg) as ds:
            if 'ClusterName' in ds.ca:
                for i in range(ds.shape[1]):
                    f = os.path.join(bigwig_dir, f'cluster_{i}.bw')
                    f2 = os.path.join(bigwig_dir, f"{ds.ca.ClusterName[i].replace(' ', '_').strip('.')}.bw")
                    os.rename(f,f2)
        subprocess.call(['rm', '-rf', 'tmp*'])
        logging.info(f'Finished saving bigwigs')

    if 'split' in config.steps:
        if not 'Karyotype' in config.steps:
            with loompy.connect(peak_agg) as ds:
                n_clusters = ds.shape[1]
                method = 'dendrogram' if (n_clusters>100) else 'coverage'
                logging.info(f'Split using {method}')

            is_split, split_job = split_subset(config, subset=name, python_exe=config.paths.pythonexe, min_RNA=config.params.min_RNA)
            if is_split:
                logging.info(f'Splitted {name}')
            else:
                logging.info(f'Splitting not succesful for {name}')
            parent = '_'.join(name.split('_')[:-1])
            old_submit = os.path.join(config.paths.build, 'submits', 'split', f"Split_{parent}.condor")
            if os.path.isfile(old_submit):
                logging.info(f'Removing {old_submit}')
                os.remove(old_submit)

    logging.info('Done with steps')