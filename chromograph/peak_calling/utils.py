import numpy as np
import sys
import os
import loompy
import multiprocessing
import logging
import pybedtools
from pybedtools import BedTool

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

def extend_fields(feature, n):
    '''
    Pads fields of a BedTool instance to n fields
    '''
    fields = feature.fields[:]
    while len(fields) < n:
        fields.append('.')
    return pybedtools.create_interval_from_list(fields)

def add_ID(feature):
    '''
    Adds a peak ID in the format of chromosome:start-end in the 4th field of a BedTool instance
    '''
    feature[3] = f'{feature[0]}:{feature[1]}-{feature[2]}'
    return feature

def add_strand(feature, strand):
    '''
    Add a strand identifier (+/-) to the 6th field of a BedTool instance
    '''
    feature[5] = strand
    return feature

def read_HOMER(file):
    '''
    Read the output of HOMER into a numpy array
    '''
    table = []
    TFs = []
    with open(file) as f:
        i = 0
        for line in f:
            if i == 0:
                cols = ['ID'] + line.split('\t')[1:]
                cols = [x.rstrip() for x in cols]
                cols = np.array([x.replace('/', '-',) for x in cols])
                clim = np.where(cols == 'GC%')[0][0] + 1
                TF_cols = [x.split(' ')[0] for x in cols[clim:]]
            if i> 0:
                table.append([x.rstrip() for x in line.split('\t')][:clim])
                tline = [x.rstrip() for x in line.split('\t')][clim:]
                tline = [x is not '' for x in tline]
                TFs.append(tline)
            i += 1

    return cols[:clim], np.array(table), TF_cols, np.array(TFs)


def reorder_by_IDs(mat: np.ndarray, IDs):
    '''
    Fast way to reorder matrix if a list of IDs with right order is available
    '''
    ## Create index dict
    idx = {k:v for v,k in enumerate(IDs)}
    
    ## Initiate empty matrix
    table = np.zeros(mat.shape, dtype=object)

    ## Populate matrix
    i = 0
    for x in range(table.shape[0]):
        table[idx[mat[x,0]],:] = mat[x,:]

    return np.array(table)

def bed_downsample(pile, level):
    '''
    '''
    p = BedTool(pile[1])
    frag_count = p.count()
    fraction = level / frag_count

    if fraction < 1:
        downsamp = p.random_subset(f=fraction)
        downsamp.saveas(pile[1])
        logging.info(f'Total fragments: {frag_count} in cluster {pile[0]}, downsampled to {downsamp.count()}')
    else:
        logging.info(f'cluster {pile[0]} was not downsampled')
    return

def Count_peaks(cells, sample_dir, f_peaks, q):
    '''
    Count peaks
    '''
    logging.info(f'Start job')
    Count_dict = {k: {} for k in cells}
    peaks = BedTool(f_peaks)  # Connect to peaks file
    i = 0

    ## Separate cells and get paths to fragment files
    for x in cells:
        
        s, c = x.split(':')
        f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
        try:
            cBed = BedTool(f) # Connect to fragment file
            pks = peaks.intersect(cBed, wa=True) # Get peaks that overlap with fragment file

            cDict = {}
            ## Extract peak_IDs
            for line in pks:
                cDict[line[3]] = 1

            Count_dict[x] = cDict
            i += 1
            if i%1000==0:
                logging.info(f'Finished counting {i} cells')
        except:
            ## If file can't be found print the path to file
            Count_dict[x] = []
            logging.info(f'Cannot find {f}')
    return q.put(Count_dict)

def Count_peaks2(cells, sample_dir, f_peaks):
    '''
    Count peaks
    '''
    logging.info(f'Start job')
    Count_dict = {k: {} for k in cells}
    peaks = BedTool(f_peaks)  # Connect to peaks file
    i = 0

    ## Separate cells and get paths to fragment files
    for x in cells:
        
        s, c = x.split(':')
        f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
        try:
            cBed = BedTool(f) # Connect to fragment file
            pks = peaks.intersect(cBed, wa=True) # Get peaks that overlap with fragment file

            cDict = {}
            ## Extract peak_IDs
            for line in pks:
                cDict[line[3]] = 1

            Count_dict[x] = cDict
            i += 1
            if i%1000==0:
                logging.info(f'Finished counting {i} cells')
        except:
            ## If file can't be found print the path to file
            Count_dict[x] = []
            logging.info(f'Cannot find {f}')
    return Count_dict

def strFrags_to_list(frags):
    '''
    Legacy function that takes np.array of fragments saved as string in loom-file and returns it as a list of fragments
    '''
    frags = frags.replace('[', '')
    frags = frags.replace(']', '')
    frags = frags.replace('"', '')
    frags = frags.replace("'", '')
    frags = frags.replace(' ', '')
    frags = frags.split(',')
    frag_list = [[frags[3*i], int(frags[3*i+1]), int(frags[3*i+2])]for i in range(int(len(frags)/3))]
    return frag_list