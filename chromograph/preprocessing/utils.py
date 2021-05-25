# Description etc to be added

import numpy as np
import os
import sys
import pybedtools
from pybedtools import BedTool
import collections
import csv
import matplotlib.pyplot as plt
import gzip
import loompy
import scipy.sparse as sparse
import json
import urllib.request
import logging
from typing import Dict
import sqlite3 as sqlite

import chromograph

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
    
    frag_dict = collections.OrderedDict()
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

def count_bins(frag_dict, barcodes, bsize):
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

        ## Retrieve dense matrix        
        data = ds[:,:].astype('int8')   
        new_data = []

        new_bins = {'chrom' : [], 'start': [], 'end': [], 'loc': []}
        sizes = []
        
        ## Loop over chromosomes to compact bins
        for i in np.unique(ds.ra.chrom):

            ## If no remainder from dividing N original bins by factor
            vals = data[ds.ra.chrom==i,:]
            if vals.shape[0]%factor == 0:
                X = rebin(vals, (int(vals.shape[0]/factor), vals.shape[1]))
            else:
                rem = vals.shape[0]%factor
                X = rebin(vals[:-rem,:], (int(vals.shape[0]/factor), vals.shape[1]))
                X2 = rebin(vals[-rem:,:], (1, vals.shape[1])) ## Merge the last (or last few) bins to one bin
                X = np.vstack((X, X2))

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
    logging.info(f"Saving fragments to separate folder for fast indexing")
    fdir = os.path.join(outdir, 'fragments')
    if not os.path.isdir(fdir):
        os.mkdir(fdir)
    if  len(os.listdir(fdir)) < len(meta['barcode']):
        i = 0
        for x in meta['barcode']:
            f = os.path.join(fdir, f'{x}.tsv.gz')
            if not os.path.exists(f):
                frags = BedTool(frag_dict[x]).filter(lambda x: x[0] in chromosomes.keys()).saveas(f)
            i += 1
            if i%1000 == 0:
                logging.info(f'Finished separating fragments for {i} cells')

    ## Count fragments inside bins
    logging.info("Count fragments overlapping with bins")
    Count_dict = count_bins(frag_dict, meta['barcode'], bsize)
    logging.info("Finished counting fragments")

    return Count_dict