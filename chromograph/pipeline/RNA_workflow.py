import loompy
import os
import subprocess
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
from chromograph.pipeline.utils import transfer_ca
from chromograph.pipeline.Add_UMAP import Add_UMAP
from chromograph.peak_calling.peak_caller import *
from chromograph.peak_calling.utils import *
from chromograph.peak_analysis.Peak_Aggregator import Peak_Aggregator
from chromograph.RNA.RNA_analysis import RNA_analysis
from chromograph.features.Generate_promoter import Generate_promoter
from chromograph.features.GA_Aggregator import GA_Aggregator
from chromograph.motifs.motif_aggregation import motif_aggregator
from chromograph.plotting.QC_plot import QC_plot

## Import punchcards
from cytograph.pipeline.punchcards import (Punchcard, PunchcardDeck, PunchcardSubset, PunchcardView)
from cytograph.clustering import PolishedLouvain
from cytograph.pipeline import Aggregator
from cytograph.species import Species

## Setup logger and load config
config = config.load_config()
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

name = sys.argv[1]

if __name__ == '__main__':

    logging.info(f'Starting post-RNA workflow')

    logging.info(f'Performing the following steps: {config.steps} for build {config.paths.build}')
    ## Check if directory exists
    if not os.path.isdir(config.paths.build):
        os.mkdir(config.paths.build)
    
    logging.info(f'Starting analysis for subset {name}')
    subset_dir = os.path.join(config.paths.build, name)
    if not os.path.isdir(subset_dir):
        os.mkdir(subset_dir)
    RNA_file = os.path.join(subset_dir, name + '_RNA.loom')
    RNA_agg = os.path.join(subset_dir, name + '_RNA.agg.loom')
    peak_file = os.path.join(subset_dir, name + '_peaks.loom')
    peak_agg = os.path.join(subset_dir, name + '_peaks.agg.loom')

    plot_dir = os.path.join(subset_dir, 'exported')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    if 'RNA' in config.steps:
        with loompy.connect(RNA_file) as ds:
            if os.path.exists(RNA_agg):
                logging.info(f'RNA aggregate file already exists')
            else:
                with loompy.connect(RNA_file) as ds:
                    Aggregator(mask=Species.detect(ds).mask(ds, ("cellcycle", "sex", "ieg", "mt"))).aggregate(ds, out_file=RNA_agg)

    if 'prom' in config.steps:
        ## Generate promoter file
        logging.info(f'Generating promoter file')
        with loompy.connect(RNA_file, 'r') as ds:
            Promoter_generator = Generate_promoter(outdir=subset_dir, poisson_pooling=config.params.poisson_pooling)
            GA_file = Promoter_generator.fit(ds)

        ## Transer column attributes
        with loompy.connect(GA_file) as ds:
            ## Aggregate GA file and annotate based on markers
            GA_agg_file = os.path.join(subset_dir, name + '_prom.agg.loom')
            Aggregator = GA_Aggregator()
            Aggregator.fit(ds, out_file=GA_agg_file)

            logging.info(f'Transferring column attributes back to peak file')
            with loompy.connect(RNA_file) as dsb:
                transfer_ca(ds, dsb, 'CellID')

    if 'peak_calling' in config.steps:
        ## Call peaks
        with loompy.connect(RNA_file, 'r+') as ds:

            if not os.path.isfile(peak_file):
                peak_caller = Peak_caller(outdir=subset_dir)
                peak_caller.fit(ds)

    ## Analyse peak-file
    if 'peak_analysis' in config.steps:
        with loompy.connect(peak_file) as ds:
            peak_aggregator = Peak_Aggregator()
            peak_aggregator.fit(ds, peak_agg, reorder=False)

        ## Plot TSNE with QC
        with loompy.connect(peak_file) as ds:
            if not 'precluster_sd' in ds.ca:
                with loompy.connect(peak_agg) as dsagg:
                    (ds.ra.precluster_mu, ds.ra.precluster_sd) = dsagg['CPM'].map((np.mean, np.std), axis=0)
            QC_plot(ds, os.path.join(plot_dir, f"{name}_peaks_TSNE_QC.png"), embedding = 'TSNE', attrs=config.params.plot_attrs)

    if 'cicero' in config.steps:
        logging.info(f'Run Cicero')
        cicero_run = os.path.join('/', *chromograph.__file__.split('/')[:-1], 'cicero', 'run_cicero.py')
        cicero_env = os.environ.copy()
        cicero_env["PATH"] = os.path.join(config.paths.cicero_path, 'bin') + ':' + cicero_env["PATH"]
        cmd = [os.path.join(config.paths.cicero_path, 'bin/python'), cicero_run, peak_file, peak_agg, 'True']
        logging.info(cmd)
        # process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=cicero_env)
        subprocess.run(cmd)
        logging.info(f'Finished running cicero')

    if 'motifs' in config.steps:
        with loompy.connect(peak_file) as ds:
            MA = motif_aggregator(name)
            MA.fit()

    ## Export bigwigs last to prevent multiprocessing error
    if 'bigwig' in config.steps:
        ## Export bigwigs by cluster
        with loompy.connect(peak_file, 'r') as ds:
            bw_dir = os.path.join(subset_dir, 'bigwigs')
            if not os.path.isdir(bw_dir):
                os.mkdir(bw_dir)
            logging.info(f'Exporting bigwigs to {bw_dir}')
            with mp.get_context().Pool(20) as pool:
                for cluster in np.unique(ds.ca.Clusters):
                    cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
                    pool.apply_async(export_bigwig, args=(cells, config.paths.samples, bw_dir, cluster,))
                pool.close()
                pool.join()
        logging.info(f'Finished saving bigwigs')