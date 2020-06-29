import numpy as np
import sys
import os
import subprocess
import loompy
import multiprocessing
import pickle as pkl
import logging
import shutil
import pybedtools
from pybedtools import BedTool
sys.path.append('/home/camiel/chromograph/')
import chromograph

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

def read_HOMER_annotation(file):
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
            if i> 0:
                table.append([x.rstrip() for x in line.split('\t')])
            i += 1

    return cols, np.array(table)

def read_HOMER_TFs(file):
    '''
    Read the output of HOMER TF annotation into a numpy array
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

    pybedtools.helpers.cleanup()
    return

def Count_peaks(id, cells, sample_dir, peak_dir):
    '''
    Count peaks
    '''
    Count_dict = {k: {} for k in cells}
    f_peaks = os.path.join(peak_dir, 'Compounded_peaks.bed')
    peaks = BedTool(f_peaks).saveas()  # Connect to peaks file, save temp to prevent io issues

    ## Separate cells and get paths to fragment files
    for x in cells:
        
        s, c = x.split(':')
        f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
        try:
            cBed = BedTool(f) # Connect to fragment file
        except:
            logging.info(f"Can't find {f}")
            logging.info(sys.exc_info()[0])
            Count_dict[x] = []
        try:
            pks = peaks.intersect(cBed, wa=True) # Get peaks that overlap with fragment file
        except:
            logging.info(f'Problem intersecting {f}')
            logging.info(sys.exc_info()[0])
            Count_dict[x] = []
            return
        try:
            cDict = {}
            ## Extract peak_IDs
            for line in pks:
                cDict[line[3]] = 1
        except:
            logging.info(f'Problem counting {f}')
            logging.info(sys.exc_info()[0])
            Count_dict[x] = []
            return
        try:
            ## Collect in output dictionary
            Count_dict[x] = cDict
        except:
            ## If file can't be found print the path to file
            Count_dict[x] = []
            logging.info(f" Problem collecting to main dict {f}")
            logging.info(sys.exc_info()[0])
            return
    logging.info(f'Completed job {id}')
    pkl.dump(Count_dict, open(os.path.join(peak_dir, f'{id}.pkl'), 'wb'))
    ## Cleanup
    pybedtools.helpers.cleanup()
    return 

def export_bigwig(cells, sample_dir, peak_dir, cluster):
    '''
    Calculates coverage for a cluster and exports as a bigwig file
    '''
    files = np.array([os.path.join(sample_dir, x[0], 'fragments', f'{x[1]}.tsv.gz') for x in cells])

    ## Check if all files exist
    ex = np.array([os.path.exists(x) for x in files])
    files = files[ex]

    fmerge = os.path.join(peak_dir, f'fragments_{cluster}.tsv.gz')
    with open(fmerge, 'wb') as out:
        for f in files:
            with open(f, 'rb') as file:
                shutil.copyfileobj(file, out)
    
    ## Downsample
    bed_downsample([cluster, fmerge], 2.5e7)
   
    ## Unzip and sort
    f_unzip = f'{fmerge.split(".")[0]}.tsv'
    f_sort = f'{fmerge.split(".")[0]}_sorted.bed'
    os.system(f'gunzip {fmerge}')
    os.system(f'sort -k 1,1 -k2,2n {f_unzip} > {f_sort}')
    os.system(f'rm {f_unzip}')

    ## Calculate coverage and scale to CPM
    cov = 1e6 / BedTool(f_sort).count()
    f_bg = f'{f_sort.split(".")[0]}.bdg'
    f_genome = os.path.join(chromograph.__path__[0], 'references/male.GRCh38.chrom.sizes')
    os.system(f'bedtools genomecov -i {f_sort} -g {f_genome} -scale {cov} -bg > {f_bg}')
    
    ## Convert to bigwig
    outfile = os.path.join(peak_dir, f'cluster_{str(cluster)}.bw')
    pybedtools.contrib.bigwig.bedgraph_to_bigwig(BedTool(f_bg), genome='hg38', output=outfile)
    
    ## Clean up
    os.system(f'rm {f_bg} {f_sort}')    
    return

def homer_motif_call(homer, f, motifs, out_file):
    '''
    Wrapper for calling HOMER annotatePeaks from python
    '''
    subprocess.call([homer, f, 'hg38', '-noann', '-nogene', '-m', motifs], stdout = open(out_file, 'wb'))
    return


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