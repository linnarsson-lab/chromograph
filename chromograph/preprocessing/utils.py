# Description etc to be added

import numpy as np
import os
import sys
import pybedtools
from pybedtools import BedTool
import collections
import csv
import matplotlib.pyplot as plt
import loompy
import pysam
import shutil
import glob
import pickle as pkl
import scipy.sparse as sparse
import json
import urllib.request
import logging
from typing import Dict
import sqlite3 as sqlite
import tempfile
import itertools
import multiprocessing as mp

import chromograph
from chromograph.pipeline.TF_IDF import TF_IDF
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def get_chrom_sizes(ref: str):
    '''
    Loads references for the sizes of the different chromosomes from the ENCODE project or UCSC website.
    Currently supports GRCh38, hg19 and mm10
    '''
    
    chrom_size = {}
    if ref in ['GRCh38', 'hg19', 'mm10']:
        with open(os.path.join(chromograph.__path__[0], 'references/male.{}.chrom.sizes'.format(ref)), 'rb') as f:
            for line in f:
                x = line.split()
                chrom_size[x[0].decode()] = int(x[1].decode())
        logging.info('Loaded chromosome sizes for {}'.format(ref))
        return chrom_size;
    else:
        logging.info('Genome not recognized')
        return

def get_blacklist(ref: str):
    '''
    Downloads bed-file containing problematic regions of the reference from the ENCODE project.
    Currently supports GRCh38, hg19 and mm10
    '''

    if ref in ['GRCh38', 'hg19', 'mm10']:
        path = os.path.join(chromograph.__path__[0], 'references/blacklist_{}.bed'.format(ref))
        logging.info('Retrieved blacklist for {}'.format(ref))
        return path
    else:
        logging.info('Genome not recognized')

def read_fragments(file):
    '''
    '''
    
    frag_dict = {} ## formerly ordered dict
    new = 0
    add = 0
    with gzip.open(file, 'rb') as f:
        next(f) ## Skip first line
        for read in f:
            if read.startswith(b'#'):
                continue
            else:
                r = read.split()
                b = r[3].decode()

                if b not in frag_dict:
                    frag_dict[b] = [[r[0].decode(), int(r[1].decode()), int(r[2].decode())]]
                    new += 1
                else:
                    frag_dict[b].append([r[0].decode(), int(r[1].decode()), int(r[2].decode())])
                    add += 1
    
        logging.info('barcodes: {}   fragments: {}'.format(new, (new+add)))
        return frag_dict;
    
def generate_bins(chrom_size, bsize, overlap:float=1):
    '''
    '''
    
    chrom_bins = collections.OrderedDict();
    i = 0
    for x in chrom_size.keys():
        for start in range(1, chrom_size[x], int(overlap*bsize)):
            end = min(start + bsize - 1, chrom_size[x]);
            bin = (x , start, end);
            chrom_bins[bin] = i;
            i += 1
    logging.info('Number of bins: {}'.format(len(chrom_bins.keys())))
    return chrom_bins;

def count_bins_dict(frag_dict, barcodes, bsize):
    '''
    '''
    
    Count_dict = collections.OrderedDict()

    i = 0
    
    for bar in barcodes:    
        if bar in frag_dict:
            frags = frag_dict[bar]
            counts = {}
            for _frag in frags:

                # If a fragment spans two bins we count it twice
                for x in set([int(int(_frag[1])/bsize)*bsize+1, int(int(_frag[2])/bsize)*bsize+1]):
                    k = (_frag[0], x, x + bsize - 1)
                    if k not in counts.keys():
                        counts[k] = 1
                    else:
                        counts[k] += 1
            Count_dict[bar] = counts
        else:
            continue
        
        i += 1
        
        if i%1000 == 0:
            logging.info(f"Finished counting {i} cells")
    
    return Count_dict;

def load_sample_metadata(path: str, sample_id: str) -> Dict[str, str]:
    '''
    From Cytograph.
    
    Args:
            path                    Path to the DB
            sample_id               Sample ID to retrieve metadata for
            
    Returns:
            result                  Dictionary containing sample metadata
    '''
    if not os.path.exists(path):
        raise ValueError(f"Samples metadata file '{path}' not found.")
    if path.endswith(".db"):
        # sqlite3
        with sqlite.connect(path) as db:
            cursor = db.cursor()
            cursor.execute("SELECT * FROM sample WHERE name = ?", (sample_id,))
            keys = [x[0].capitalize() for x in cursor.description]
            vals = cursor.fetchone()
            if vals is not None:
                return dict(zip(keys, vals))
            raise ValueError(f"SampleID '{sample_id}' was not found in the samples database.")
    else:
        result = {}
        with open(path) as f:
            headers = [x.lower() for x in f.readline()[:-1].split("\t")]
            if "sampleid" not in headers and 'name' not in headers:
                raise ValueError("Required column 'SampleID' or 'Name' not found in sample metadata file")
            if "sampleid" in headers:
                sample_metadata_key_idx = headers.index("sampleid")
            else:
                sample_metadata_key_idx = headers.index("name")
            sample_found = False
            for line in f:
                items = line[:-1].split("\t")
                if len(items) > sample_metadata_key_idx and items[sample_metadata_key_idx] == sample_id:
                    for i, item in enumerate(items):
                        result[headers[i]] = item
                    sample_found = True
        if not sample_found:
            raise ValueError(f"SampleID '{sample_id}' not found in sample metadata file")
        return result

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).sum(-1).sum(1)

def mergeBins(f, bin_size):
    with loompy.connect(f, 'r') as ds:
        ## Assume that original file has bin sizes of size 5kb
        factor = int(bin_size/5000)

        ## Setup lists      
        new_data = []
        new_bins = {'chrom' : [], 'start': [], 'end': [], 'loc': []}
        sizes = []
        
        ## Loop over chromosomes to compact bins
        for i in np.unique(ds.ra.chrom):

            ## If no remainder from dividing N original bins by factor
            vals = ds[ds.ra.chrom==i,:].astype('int8')
            if vals.shape[0]%factor == 0:
                X = rebin(vals, (int(vals.shape[0]/factor), vals.shape[1]))
            elif vals.shape[0] > factor:
                rem = vals.shape[0]%factor
                X = rebin(vals[:-rem,:], (int(vals.shape[0]/factor), vals.shape[1]))
                X2 = rebin(vals[-rem:,:], (1, vals.shape[1])) ## Merge the last (or last few) bins to one bin
                X = np.vstack((X, X2))
            else:
                X = np.sum(vals, axis=0)

            new_data.append(X.astype('int8'))

            for start, end in zip(ds.ra.start[ds.ra.chrom==i][::factor], ds.ra.end[ds.ra.chrom==i][(factor-1)::factor]):
                new_bins['chrom'].append(i)
                new_bins['start'].append(start)
                new_bins['end'].append(end)
                new_bins['loc'].append(f'{i}:{start}:{end}')

            ## If there was a remainder, name of last bin will be the added to the dictionary
            if len(ds.ra.end[ds.ra.chrom==i][(factor-1)::factor]) < X.shape[0]:
                start = str(int(new_bins['end'][-1]) + 1)
                end = np.max(ds.ra.end[ds.ra.chrom==i].astype('int'))
                new_bins['chrom'].append(i)
                new_bins['start'].append(start)
                new_bins['end'].append(end)
                new_bins['loc'].append(f'{i}:{start}:{end}')

        ## Make matrix sparse
        matrix = sparse.coo_matrix(np.vstack(new_data)).tocsr()
        
        ## Create loomfile
        sampleid = f.split('/')[-2] + '_' + str(int(bin_size/1000)) + 'kb'
        floom = os.path.join(os.path.dirname(f), sampleid + '.loom')
        
        loompy.create(filename=floom, 
                      layers=matrix, 
                      row_attrs=new_bins, 
                      col_attrs=ds.ca,
                      file_attrs=ds.attrs)
        
        ## Change bin_size in attributes
        with loompy.connect(floom) as dsout:
            dsout.attrs['bin_size'] = bin_size
        
        logging.info(f"Loom-file with {str(int(bin_size/1000)) + 'kb'} bins saved as {floom}")

def fragments_to_count(x):
    '''
    '''

    ff, outdir, meta, bsize, chromosomes = x

    ## Read Fragments and generate size bins
    logging.info("Read fragments into dict")
    frag_dict = read_fragments(ff)

    ## Split fragments to seperate files for fast indexing
    fdir = os.path.join(outdir, 'fragments')
    if not os.path.isdir(fdir):
        os.mkdir(fdir)
    if  len(os.listdir(fdir)) < len(meta['barcode']):
        logging.info(f"Saving fragments to separate folder for fast indexing")
        i = 0
        for x in meta['barcode']:
            f = os.path.join(fdir, f'{x}.tsv.gz')
            if not os.path.exists(f):
                frags = BedTool(frag_dict[x]).filter(lambda x: x[0] in chromosomes.keys()).saveas(f)
            i += 1
            if i%1000 == 0:
                logging.info(f'Finished separating fragments for {i} cells')
                pybedtools.helpers.cleanup() ## Do some intermittent cleanup

    ## Count fragments inside bins
    logging.info("Count fragments overlapping with bins")
    Count_dict = count_bins(frag_dict, meta['barcode'], bsize)
    logging.info("Finished counting fragments")

    pkl.dump(Count_dict, open(os.path.join(outdir, 'counts.pkl'), 'wb'))
    pybedtools.helpers.cleanup()

    return

def split_fragments(ff, outdir, meta, chromosomes):
    '''
    '''
    fdir = os.path.join(outdir, 'fragments')
    if not os.path.isdir(fdir):
        os.mkdir(fdir)
    if  len(os.listdir(fdir)) < len(meta['barcode']):
        ## Read Fragments
        logging.info("Read fragments into dict")
        frag_dict = read_fragments(ff)
        logging.info(f"Saving fragments to separate folder for fast indexing")
        i = 0
        try:
            for x in meta['barcode']:
                f = os.path.join(fdir, f'{x}.tsv.gz')
                if not os.path.exists(f):
                    frags = BedTool(frag_dict[x]).filter(lambda x: x[0] in chromosomes.keys()).saveas(f)
                i += 1
                if i%1000 == 0:
                    logging.info(f'Finished separating fragments for {i} cells')
                    pybedtools.helpers.cleanup() ## Do some intermittent cleanup 
        except:
            pybedtools.helpers.cleanup()
    else:
        logging.info(f'Fragments already split')

def save_fragments_to_file(frag_dict, outdir):
    try:
        for cell in frag_dict:
            f = os.path.join(outdir, f'{cell}.bed')
            with open(f, 'a') as file:
                writer = csv.writer(file, delimiter='\t')
                for line in frag_dict[cell]:  
                    line =  [str(x) for x in line]
                    writer.writerow(line)
    
    except Exception as e:
        logging.info(f'failed {f}')
        logging.info(e)
        return

def bed_to_zip(files):
    try:
        for f in files:
            f_out = f"{f.split('.')[0]}.tsv.gz"
            BedTool(f).remove_invalid().saveas(f_out)
    except:
        logging.info(f'failed {f}')
        pybedtools.helpers.cleanup()

def split_fragments2(ff, sample_dir, meta, chromosomes):
    ## Check which chromosomes are present
    logging.info(f'Checking if all chromosomes are present') ## tabix error when iterating over missing chromosome (chrM)
    tbx = pysam.TabixFile(ff)
    present = set()
    for row in tbx.fetch(parser=pysam.asBed()):
        if row[0] not in present:
            present.add(row[0])
    present = [x for x in present if x in chromosomes.keys()]

    ## Empty existing fragment directory
    outdir = os.path.join(sample_dir, 'fragments')
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)

    ## Process files
    logging.info(f'Start processing data')
    bars = set(meta['barcode'])
    for chr in sorted(present):
        frag_dict = {}

        ## Retrieve reads
        for row in tbx.fetch(chr, parser=pysam.asBed()):
            if row[3] in bars:
                if row[3] not in frag_dict:
                    frag_dict[row[3]] = [row[:3]]
                else:
                    frag_dict[row[3]].append(row[:3])

        ## Write to file    
        logging.info(f'Processing {chr}, N cells: {len(frag_dict.keys())}')
        chunks = np.array_split(np.array([x for x in frag_dict.keys()]), mp.cpu_count())
        chunks = [{k:frag_dict[k] for k in chunk} for chunk in chunks]   
        with mp.get_context().Pool() as pool:
            for ck in chunks:
                pool.apply_async(save_fragments_to_file, (ck, outdir,))
            pool.close()
            pool.join()

    ## Convert to tsv.gz and remove malformed lines
    logging.info(f'Converting files to tsv.gz')
    chunks = np.array_split(np.array(glob.glob(f"{outdir}/{'*.bed'}")), mp.cpu_count())
    with mp.get_context().Pool() as pool:
        for ck in chunks:
            pool.apply_async(bed_to_zip, (ck,))
        pool.close()
        pool.join()    
    
    ## Cleanup
    for file in glob.glob(f"{outdir}/{'*.bed'}"):
        os.remove(file) 
    return


def Count_bins(id, cells, sample_dir, chrom_bins, verbose: bool = False):
    '''
    Count bins
    Args:
    '''
    chrom_dict = {f'{k[0]}:{k[1]}-{k[2]}':v for k,v in chrom_bins.items()}
    bins = [list(x) for x in chrom_bins.keys()]
    for i, x in enumerate(bins):
        bins[i].append(f'{x[0]}:{x[1]}-{x[2]}')
    bins = BedTool(bins).saveas()  # Connect to peaks file, save temp to prevent io issues
    mat = sparse.lil_matrix((bins.count(),len(cells)), dtype='int8')
    ## Separate cells and get paths to fragment files
    try:    
        for i, c in enumerate(cells):
            f = os.path.join(sample_dir, 'fragments', f'{c}.tsv.gz')
            cBed = BedTool(f).sort() # Connect to fragment file, make sure it's sorted to prevent 'invalid interval error'
            pks = bins.intersect(cBed, wa=True) # Get bins that overlap with fragment file
            ## Extract peak_IDs
            for line in pks:
                k = line[3]
                ## Add count to dict
                mat[chrom_dict[k],i] += 1
    except Exception as e:
        logging.info(f'Error in {f}')
        logging.info(e)
        pybedtools.helpers.cleanup()
    ## Cleanup
    pybedtools.helpers.cleanup()
    pkl.dump(mat.tocsc(), open(os.path.join(sample_dir, f'{id}.pkl'), 'wb'))
    return

def add_TSNE(ds):
    ## Use only Q25 top bins
    logging.info('Adding TSNE')
    logging.info(f'Calculating row wise nonzero rate')
    NCells = ds.map([np.count_nonzero], axis=0)[0]
    q = np.quantile(NCells, .75)
    logging.info(f'Using only bins present in more than {q} out of {ds.shape[1]} cells')
    valid = NCells > q

    f_temp = ds.filename + '.tmp'
    if os.path.isfile(f_temp):
        os.remove(f_temp)
    logging.info(f'Making temp file')

    with loompy.new(f_temp) as dst:
        x = np.where(valid)[0]
        for (ix, selection, view) in tqdm(ds.scan(layers = [''], axis=1)):
            dst.add_columns(view[''][x,:], col_attrs=view.ca, row_attrs={'ID': ds.ra.ID[x]})
        dst.ra.Valid = np.ones(dst.shape[0])

        ## Term-Frequence Inverse-Data-Frequency ##
        logging.info(f'Performing TF-IDF')
        tf_idf = TF_IDF(layer='')
        tf_idf.fit(dst)
        dst.layers['TF-IDF'] = 'float16'
        progress = tqdm(total=dst.shape[1])
        for (_, selection, view) in dst.scan(axis=1, batch_size=512):
            dst['TF-IDF'][:,selection] = tf_idf.transform(view[:,:], selection)
            progress.update(512)
        progress.close()
        logging.info(f'Finished fitting TF-IDF')

        ## Fit PCA
        logging.info(f'Fitting PCA')
        pca = PCA(n_components=40).fit_transform(dst['TF_IDF'][:,:].T)
        dst.ca.PCA = pca

        ## Add TSNE
        xy = TSNE(angle=.5, perplexity=30, verbose=False).fit_transform(dst.ca.PCA)
        ds.ca.TSNE = xy

    os.remove(dst)
    return
