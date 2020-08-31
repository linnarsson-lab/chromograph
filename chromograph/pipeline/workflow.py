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
sys.path.append('/home/camiel/chromograph/')
import chromograph
from chromograph.pipeline.Bin_analysis import *
from chromograph.pipeline import config
from chromograph.peak_analysis.peak_analysis import Peak_analysis
from chromograph.pipeline.utils import transfer_ca
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

## Import punchcards
from cytograph.pipeline.punchcards import (Punchcard, PunchcardDeck, PunchcardSubset, PunchcardView)

## Setup logger and load config
config = config.load_config()
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class Peak_caller:
    def __init__(self, outdir) -> None:
        """
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:
            ds                    Loom connection

        Remarks:
        
        """
        self.config = chromograph.pipeline.config.load_config()
        self.outdir = outdir
        self.peakdir = os.path.join(outdir, 'peaks')
        self.loom = ''
        self.precomp = 'None'
        logging.info("Peak Caller initialised")
    
    def fit(self, ds: loompy.LoomConnection) -> None:
        '''
        Generate fragments piles based on cluster identities and use MACS to call peaks
        
        Args:

        Returns:
        
        Remarks:
        
        '''
        ## Get sample name from loom-file
        name = ds.filename.split(".")[0]

        ## Check if location for peaks and compounded fragments exists
        if not os.path.isdir(self.peakdir):
            os.mkdir(self.peakdir)   

        ## Check if Compounded peaks already exists
        if not os.path.exists(os.path.join(self.peakdir, 'Compounded_peaks.bed')):
            path_precomp = os.path.join(self.config.paths.build, 'All', 'peaks', 'Compounded_peaks.bed')
            if os.path.exists(path_precomp):
                logging.info(f'Use peaks computed for full dataset')
                self.precomp = 'All'
                shutil.copyfile(path_precomp, os.path.join(self.peakdir, 'Compounded_peaks.bed'))
                f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
            
                path_pre_annot = os.path.join(self.config.paths.build, 'All', 'peaks', 'annotated_peaks.txt')
                if os.path.exists(path_pre_annot):
                    logging.info(f'Use annotation of precomputed peaks')
                    shutil.copyfile(path_pre_annot, os.path.join(self.peakdir, 'annotated_peaks.txt'))
                    shutil.copyfile(os.path.join(self.config.paths.build, 'All', 'peaks', 'motif_annotation.txt'), os.path.join(self.peakdir, 'motif_annotation.txt'))

                # ## Generate bigwigs
                # pool = mp.Pool(20)
                # logging.info('Exporting bigwigs')
                # for cluster in tqdm(np.unique(ds.ca.Clusters)):
                #     cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
                #     pool.apply_async(export_bigwig, args=(cells, self.config.paths.samples, self.peakdir, cluster,))
                # pool.close()
                # pool.join()
            
            else:
                logging.info('No precomputed peak list. Calling peaks')
                logging.info(f'Saving peaks to folder {self.peakdir}')

                chunks = []
                for i in np.unique(ds.ca['Clusters']):
                    cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == i]]
                    files = [os.path.join(self.config.paths.samples, x[0], 'fragments', f'{x[1]}.tsv.gz') for x in cells]
                    if len(cells) > self.config.params.peak_min_cells:
                        chunks.append([i,files])

                logging.info('Start merging fragments by cluster')
                piles = []
                for ck in chunks:
                    files = np.array(ck[1])
                    ex = np.array([os.path.exists(x) for x in files])
                    files = files[ex]

                    fmerge = os.path.join(self.peakdir, f'cluster_{ck[0]}.tsv.gz')
                    with open(fmerge, 'wb') as out:
                        for f in files:
                            with open(f, 'rb') as file:
                                shutil.copyfileobj(file, out)
                    piles.append([ck[0], fmerge])
                    logging.info(f'Finished with cluster {ck[0]}')

                logging.info(f'Downsample pile-ups to {self.config.params.peak_depth / 1e6} million fragments')
                pool = mp.Pool(20) 
                for pile in piles:
                    pool.apply_async(bed_downsample, args=(pile, self.config.params.peak_depth,))
                pool.close()
                pool.join()

                ## Manually call garbage collection to prevent memory leaks from mp
                gc.collect()

                logging.info(f'Finished downsampling')

                logging.info(f'Start calling peaks')
                pool = mp.Pool(20,maxtasksperchild=1) 
                for pile in piles:
                    pool.apply_async(call_MACS, args=(pile, self.peakdir, self.config.paths.MACS,))
                pool.close()
                pool.join()

                ## Manually call garbage collection to prevent memory leaks from mp
                gc.collect()
                    
                ## Compound the peak lists
                peaks = [BedTool(x) for x in glob.glob(os.path.join(self.peakdir, '*.narrowPeak'))]
                logging.info('Identified on average {} peaks per cluster'.format(np.int(np.mean([len(x) for x in peaks]))))
                peaks_all = peaks[0].cat(*peaks[1:])
                peaks_all.merge()

                ## Substract blacklist
                black_list = BedTool(get_blacklist(self.config.params.reference_assembly))
                peaks_all.subtract(black_list, A=True)

                ## Pad out and save peaks
                f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
                peaks_all = peaks_all.each(extend_fields, 6).each(add_ID).each(add_strand, '+').saveas(f)   ## Pad out the BED-file and save
                logging.info(f'Identified {peaks_all.count()} peaks after compounding list')

                ## Clean up
                for file in glob.glob(os.path.join(self.peakdir, '*.tsv.gz')):
                    os.system(f'rm {file}')
                for file in glob.glob(os.path.join(self.peakdir, '*.narrowPeak')):
                    os.system(f'rm {file}')
        else:
            logging.info('Compounded peak file already present, loading now')
            f = os.path.join(self.peakdir, 'Compounded_peaks.bed')
            peaks_all = BedTool(f)

            # ## Generate bigwigs
            # pool = mp.Pool(20)
            # logging.info('Exporting bigwigs')
            # for cluster in tqdm(np.unique(ds.ca.Clusters)):
            #     cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
            #     pool.apply_async(export_bigwig, args=(cells, self.config.paths.samples, self.peakdir, cluster,))
            # pool.close()
            # pool.join()

        ## Check All_peaks.loom exists, get subset
        all_peaks_loom = os.path.join(self.config.paths.build, 'All', 'All_peaks.loom')
        if os.path.exists(all_peaks_loom):
            logging.info(f'Main peak matrix already exists')
            with loompy.connect(all_peaks_loom) as dsp:
                selection = np.array([x in ds.ca.CellID for x in dsp.ca.CellID])
                
            self.loom = os.path.join(self.outdir, f'{name}_peaks.loom')
            loompy.combine_faster([all_peaks_loom], self.loom, selections=[selection], key = 'ID')
            logging.info(f'Finished creating peak file')

            with loompy.connect(self.loom, 'r+') as ds_out:
                logging.info(f'Transferring attributes')
                transfer_ca(ds, ds_out, 'CellID')
                return
        
        f_annot = os.path.join(self.peakdir, 'annotated_peaks.txt')
        if not os.path.exists(f_annot):
            ## Annotate peaks
            logging.info(f'Annotating peaks')
            homer = os.path.join(self.config.paths.HOMER, 'annotatePeaks.pl')
            genes = os.path.join(self.config.paths.ref, 'genes', 'genes.gtf')
            motifs = os.path.join(chromograph.__path__[0], 'references/human_TFs.motifs') ## Read the motif file from chromograph reference
            cmd = f'{homer} {f} hg38 -gtf {genes} -CpG > {f_annot}'  ## Command to call HOMER
            os.system(cmd)  ## Actually call HOMER

            ## Annotate motifs
            split_dir = os.path.join(self.peakdir, 'compound_split')
            out_motifs = os.path.join(self.peakdir, 'out_motifs')
            [os.mkdir(x) for x in [split_dir, out_motifs] if not os.path.exists(x)]
            subprocess.run(['awk', '{print $0 >>' + f'"{split_dir}/"' + '$1".bed"}', os.path.join(self.peakdir, 'Compounded_peaks.bed')]) ## Split by chromosome since motif annotation takes a lot of RAM

            pool = mp.Pool(5,maxtasksperchild=1)
            logging.info('Annotating motifs')
            for file in tqdm(os.listdir(split_dir)):
                chrom_file = os.path.join(split_dir, file)
                out_file = os.path.join(out_motifs, f'{file.split("/")[-1].split(".")[0]}.txt')
                pool.apply_async(homer_motif_call, args=(homer,chrom_file, motifs, out_file,))
            pool.close()
            pool.join()

            ## Merge motif outputs
            motif_outputs = [os.path.join(out_motifs, x) for x in os.listdir(out_motifs)]
            motif_annotation = open(os.path.join(self.peakdir, 'motif_annotation.txt'), 'a')
            subprocess.call(['head', '-1', motif_outputs[0]], stdout = motif_annotation)
            for x in motif_outputs:
                subprocess.call(['tail', '-n', '+2', '-q',x], stdout = motif_annotation)

        ## Load and reorder HOMER output
        logging.info(f'Reordering annotation file')
        cols, table = read_HOMER_annotation(f_annot)
        peak_IDs = np.array([x[3] for x in peaks_all])
        table = reorder_by_IDs(table, peak_IDs)
        annot = {cols[i]: table[:,i] for i in range(table.shape[1])}
        logging.info('Plotting peak annotation wheel')
        plot_peak_annotation_wheel(annot, os.path.join(self.outdir, 'exported', 'peak_annotation_wheel.png'))

        logging.info(f'Start counting peaks')
        pool = mp.Pool(20, maxtasksperchild=1)
        chunks = np.array_split(ds.ca['CellID'], np.int(np.ceil(ds.shape[1]/1000)))
        for i, cells in enumerate(chunks):
            pool.apply_async(Count_peaks, args=(i, cells, self.config.paths.samples, self.peakdir, os.path.join(self.peakdir, 'Compounded_peaks.bed'), ))
        pool.close()
        pool.join()
        
        # Order dict for rows
        r_dict = {k: v for v,k in enumerate(annot['ID'])} 

        logging.info("Generating Sparse matrix")
        col = []
        row = []
        v = []

        cix = 0
        IDs = []
        dict_files = glob.glob(os.path.join(self.peakdir, '*.pkl'))
        for file in dict_files:
            Counts = pkl.load(open(file, 'rb'))
            for cell in Counts:
                if len(Counts[cell]) > 0:
                    for key in (Counts[cell]):
                        col.append(cix)
                        row.append(r_dict[key])
                        v.append(np.int8(Counts[cell][key]))
                    cix+=1
                    IDs.append(cell)
        logging.info(f'CellID order is maintained: {np.array_equal(ds.ca.CellID, np.array(IDs))}')
        matrix = sparse.coo_matrix((v, (row,col)), shape=(len(r_dict.keys()), len(ds.ca['CellID']))).tocsc()
        logging.info(f'Matrix has shape {matrix.shape} with {matrix.nnz} elements')

        ## Create loomfile
        logging.info("Constructing loomfile")
        self.loom = os.path.join(self.outdir, f'{name}_peaks.loom')

        loompy.create(filename=self.loom, 
                    layers=matrix, 
                    row_attrs=annot, 
                    col_attrs={'CellID': np.array(IDs)},
                    file_attrs=dict(ds.attrs))
        logging.info(f'Transferring column attributes')
        with loompy.connect(self.loom) as ds2:
            ds2.attrs['peak_file'] = self.precomp
            transfer_ca(ds, ds2, 'CellID')
        logging.info(f'Loom peaks file saved as {self.loom}')

        for file in glob.glob(os.path.join(self.peakdir, '*.pkl')):
            os.system(f'rm {file}')

        ## Clean up stranded pybedtools tmp files
        pybedtools.helpers.cleanup(verbose=True, remove_all=True)
        return self.loom

if __name__ == '__main__':

    bsize = f'{int(config.params.bin_size/1000)}kb'

    ## Load punchcard and setup decks for analysis
    deck = PunchcardDeck(config.paths.build)
    for subset in deck.get_leaves():
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
                    good_cells = (ds.ca.DoubletFinderFlag == 0) & (ds.ca.passed_filters > 5000) & (ds.ca.passed_filters < 1e5) & (ds.ca.promoter_region_fragments/ds.ca.passed_filters > config.params.FRIP)
                    selections.append(good_cells)

            # ## Merge Bin files
            if not os.path.exists(binfile):
                logging.info(f'Input samples {samples}')
                loompy.combine_faster(inputfiles, binfile, selections=selections, key = 'loc', skip_attrs=config.params.skip_attrs)
                # loompy.combine(inputfiles, outfile, key = 'loc')       ## Use if running into memory errors
                logging.info('Finished combining loom-files')
            else:
                logging.info('Combined bin file already exists, using this for analysis')

            ## Run primary Clustering and embedding
            with loompy.connect(binfile, 'r+') as ds:
                bin_analysis = Bin_analysis(outdir=subset_dir)
                bin_analysis.fit(ds)

        if 'peak_calling' in config.steps:
            ## Call peaks
            with loompy.connect(binfile, 'r+') as ds:
                peak_caller = Peak_caller(outdir=subset_dir)
                peak_caller.fit(ds)

        if 'peak_analysis' in config.steps:
            ## Analyse peak-file
            peak_file = os.path.join(subset_dir, name + '_peaks.loom')
            peak_agg = os.path.join(subset_dir, name + '_peaks.agg.loom')
            with loompy.connect(peak_file, 'r+') as ds:
                peak_analysis = Peak_analysis(outdir=subset_dir)
                peak_analysis.fit(ds)

                peak_aggregator = Peak_Aggregator()
                peak_aggregator.fit(ds, peak_agg)

                logging.info(f'Transferring column attributes and column graphs back to bin file')
                with loompy.connect(binfile) as dsb:
                    transfer_ca(ds, dsb, 'CellID')

                ## Export bigwigs by cluster
                pool = mp.Pool(20)
                logging.info('Exporting bigwigs')
                for cluster in tqdm(np.unique(ds.ca.Clusters)):
                    cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
                    pool.apply_async(export_bigwig, args=(cells, config.paths.samples, os.path.join(subset_dir, 'peaks'), cluster,))
                pool.close()
                pool.join()


        if 'GA' in config.steps:
            ## Generate promoter file
            with loompy.connect(binfile, 'r') as ds:
                Promoter_generator = Generate_promoter(outdir=subset_dir)
                GA_file = Promoter_generator.fit(ds)

            # inputfiles = [os.path.join(config.paths.samples, '10X' + sample, f'10X{sample}_GA.loom') for sample in samples]

            # ## Check if cells have been selected
            # selections = []
            # with loompy.connect(binfile, 'r') as ds:
            #     IDs = set(ds.ca.CellID)
            #     for f in inputfiles:
            #         with loompy.connect(f, 'r') as dsg:
            #             selections.append(np.array([x in IDs for x in dsg.ca.CellID]))

            # if not os.path.exists(GA_file):
            #     logging.info(f'Combining GA looms')
            #     loompy.combine_faster(inputfiles, GA_file, selections=selections, key = 'Accession', skip_attrs=config.params.skip_attrs)

            ## Transer column attributes
            with loompy.connect(GA_file) as ds:
                # logging.info(f'Transferring column attributes and column graphs to GA file')
                # with loompy.connect(binfile) as dsb:
                #     transfer_ca(dsb, ds, 'CellID')
                # ## Smoooth over NN graph
                # Smooth = GeneSmooth()
                # Smooth.fit(ds)

                ## Aggregate GA file and annotate based on markers
                GA_agg_file = os.path.join(subset_dir, name + '_GA.agg.loom')
                Aggregator = GA_Aggregator()
                Aggregator.fit(ds, out_file=GA_agg_file)

                logging.info(f'Transferring column attributes back to bin file')
                with loompy.connect(binfile) as dsb:
                    transfer_ca(ds, dsb, 'CellID')

        if 'motifs' in config.steps:
            if 'peak_file' not in locals():
                peak_file = os.path.join(subset_dir, name + '_peaks.loom')

            with loompy.connect(peak_file) as ds:
                motif_compounder = Motif_compounder(outdir=subset_dir)
                motif_compounder.fit(ds)