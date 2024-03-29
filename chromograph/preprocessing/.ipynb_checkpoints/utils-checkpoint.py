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
        for read in f:
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
    
def generate_bins(chrom_size, bsize):
    '''
    '''
    
    chrom_bins = collections.OrderedDict();
    i = 0
    for x in chrom_size.keys():
        for start in range(1, chrom_size[x], bsize):
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