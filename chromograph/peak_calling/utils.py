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
import glob
import traceback
import chromograph
from scipy import sparse

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

def bed_downsample(pile, level, verbose: bool = False):
    '''
    '''
    try:
        p = BedTool(pile[1])
        frag_count = p.count()
        fraction = level / frag_count

        if fraction < 1:
            downsamp = p.random_subset(f=fraction)
            downsamp.saveas(pile[1])
            if verbose:
                logging.info(f'Total fragments: {frag_count} in cluster {pile[0]}, downsampled to {downsamp.count()}')
        else:
            if verbose:
                logging.info(f'cluster {pile[0]} was not downsampled')

        if verbose:
            logging.info(f'Exported cluster {pile[0]}')
        pybedtools.helpers.cleanup()
    except:
        pybedtools.helpers.cleanup()
    return

def Count_peaks(id, cells, sample_dir, peak_dir, f_peaks, ref_type: str = 'peaks', verbose: bool = False):
    '''
    Count peaks

    Args:

    type                peaks or genes
    '''
    Count_dict = {k: {} for k in cells}
    peaks = BedTool(f_peaks).saveas()  # Connect to peaks file, save temp to prevent io issues

    ## Separate cells and get paths to fragment files
    try:
        for x in cells:
        
            s, c = x.split(':')
            f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
            f2 = os.path.join(sample_dir, s, 'fragments', f'{c}-1.tsv.gz')
            if os.path.exists(f):
                cBed = BedTool(f).sort() # Connect to fragment file, make sure it's sorted to prevent 'invalid interval error'
            else:
                cBed = BedTool(f2).sort()
            if ref_type == 'peaks':
                pks = peaks.intersect(cBed, wa=True) # Get peaks that overlap with fragment file
            elif ref_type == 'genes':
                pks = peaks.intersect(cBed, wa=True, wb=True) # Get genes that overlap with fragment file
            
            ## Extract peak_IDs
            cDict = {}
            for line in pks:
                if ref_type == 'peaks':
                    k = line[3]
                elif ref_type == 'genes':
                    k = line.attrs['gene_id']
                ## Add count to dict
                if k not in cDict.keys():
                    cDict[k] = 1
                else:
                    cDict[k] += 1
            ## Collect in output dictionary
            Count_dict[x] = cDict
    except:
        pybedtools.helpers.cleanup()
        return

    # for x in cells:
        
    #     s, c = x.split(':')
    #     f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
    #     f2 = os.path.join(sample_dir, s, 'fragments', f'{c}-1.tsv.gz')
    #     try:
    #         if os.path.exists(f):
    #             cBed = BedTool(f).sort() # Connect to fragment file, make sure it's sorted to prevent 'invalid interval error'
    #         else:
    #             cBed = BedTool(f2).sort()
    #     except:
    #         logging.info(f"Can't find {f}")
    #         logging.info(traceback.format_exc())
    #         Count_dict[x] = []
    #     try:
    #         if ref_type == 'peaks':
    #             pks = peaks.intersect(cBed, wa=True) # Get peaks that overlap with fragment file
    #         elif ref_type == 'genes':
    #             pks = peaks.intersect(cBed, wa=True, wb=True) # Get genes that overlap with fragment file
    #     except:
    #         logging.info(f'Problem intersecting {f}')
    #         logging.info(traceback.format_exc())
    #         Count_dict[x] = []
    #         return
    #     try:
    #         cDict = {}
    #         ## Extract peak_IDs
    #         for line in pks:
    #             if ref_type == 'peaks':
    #                 k = line[3]
    #             elif ref_type == 'genes':
    #                 k = line.attrs['gene_id']
    #             ## Add count to dict
    #             if k not in cDict.keys():
    #                 cDict[k] = 1
    #             else:
    #                 cDict[k] += 1
    #     except:
    #         logging.info(f'Problem counting {f}')
    #         logging.info(traceback.format_exc())
    #         Count_dict[x] = []
    #         return
    #     try:
    #         ## Collect in output dictionary
    #         Count_dict[x] = cDict
    #     except:
    #         ## If file can't be found print the path to file
    #         Count_dict[x] = []
    #         logging.info(f" Problem collecting to main dict {f}")
    #         logging.info(traceback.format_exc())
    #         return
    if verbose:
        logging.info(f'Completed job {id}')
    pkl.dump(Count_dict, open(os.path.join(peak_dir, f'{id}.pkl'), 'wb'))
    ## Cleanup
    pybedtools.helpers.cleanup()
    return 

def Count_peaks_matrix(id, cells, sample_dir, peak_dir, f_peaks, ref_type: str = 'peaks', verbose: bool = False):
    '''
    Count peaks

    Args:

    type                peaks or genes
    '''
    peaks = BedTool(f_peaks).saveas()  # Connect to peaks file, save temp to prevent io issues
    peak_dict = {p: i for i, p in enumerate([x[3] for x in peaks])}
    mat = sparse.lil_matrix((peaks.count(),len(cells)), dtype='int8')

    ## Separate cells and get paths to fragment files
    try:
        for i, x in enumerate(cells):
            s, c = x.split(':')
            f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
            f2 = os.path.join(sample_dir, s, 'fragments', f'{c}-1.tsv.gz')
            if os.path.exists(f):
                cBed = BedTool(f).sort() # Connect to fragment file, make sure it's sorted to prevent 'invalid interval error'
            else:
                cBed = BedTool(f2).sort()
            if ref_type == 'peaks':
                pks = peaks.intersect(cBed, wa=True) # Get peaks that overlap with fragment file
                for line in pks:
                    k = line[3]
                    mat[peak_dict[k],i] += 1
            elif ref_type == 'genes':
                pks = peaks.intersect(cBed, wa=True, wb=True) # Get genes that overlap with fragment file
                for line in pks:
                    k = line.attrs['gene_id']
                    mat[peak_dict[k],i] += 1

            # ## Extract peak_IDs
            # for line in pks:
            #     if ref_type == 'peaks':
            #         k = line[3]
            #     elif ref_type == 'genes':
            #         k = line.attrs['gene_id']
            #     ## Add count to dict
            #     mat[peak_dict[k],i] += 1
    except Exception as e:
        logging.info(f'{id} failed!')
        logging.info(e)
        pybedtools.helpers.cleanup()
        return

    # for i, x in enumerate(cells):
    #     s, c = x.split(':')
    #     f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
    #     f2 = os.path.join(sample_dir, s, 'fragments', f'{c}-1.tsv.gz')
    #     try:
    #         if os.path.exists(f):
    #             cBed = BedTool(f).sort() # Connect to fragment file, make sure it's sorted to prevent 'invalid interval error'
    #         else:
    #             cBed = BedTool(f2).sort()
    #     except:
    #         logging.info(f"Can't find {f}")
    #         logging.info(traceback.format_exc())
    #     try:
    #         if ref_type == 'peaks':
    #             pks = peaks.intersect(cBed, wa=True) # Get peaks that overlap with fragment file
    #         elif ref_type == 'genes':
    #             pks = peaks.intersect(cBed, wa=True, wb=True) # Get genes that overlap with fragment file
    #     except:
    #         logging.info(f'Problem intersecting {f}')
    #         logging.info(traceback.format_exc())
    #         return
    #     try:
    #         ## Extract peak_IDs
    #         for line in pks:
    #             if ref_type == 'peaks':
    #                 k = line[3]
    #             elif ref_type == 'genes':
    #                 k = line.attrs['gene_id']
    #             ## Add count to dict
    #             mat[peak_dict[k],i] += 1
                
    #     except:
    #         logging.info(f'Problem counting {f}')
    #         logging.info(traceback.format_exc())
    #         return
    if verbose:
        logging.info(f'Completed job {id}')

    ## Cleanup
    pybedtools.helpers.cleanup()
    return mat.tocsc()

def export_bigwig(cells, sample_dir, peak_dir, cluster, verbose=False):
    '''
    Calculates coverage for a cluster and exports as a bigwig file
    '''
    files = [glob.glob(os.path.join(sample_dir, x[0], 'fragments', f'{x[1]}*.tsv.gz')) for x in cells]
    files = [x for s in files for x in s]

    if verbose:
        logging.info(f'Found {len(files)} files')
    fmerge = os.path.join(peak_dir, f'fragments_{cluster}.tsv.gz')
    with open(fmerge, 'wb') as out:
        for f in files:
            with open(f, 'rb') as file:
                shutil.copyfileobj(file, out)
    
    ## Downsample
    bed_downsample([cluster, fmerge], 2.5e7)
   
    if verbose:
        logging.info('Bed downsampled')
    ## Unzip and sort
    f_unzip = f'{fmerge.split(".")[0]}.tsv'
    f_sort = f'{fmerge.split(".")[0]}_sorted.bed'
    os.system(f'gunzip {fmerge}')
    os.system(f'sort -k 1,1 -k2,2n {f_unzip} > {f_sort}')
    os.system(f'rm {f_unzip}')

    if verbose:
        logging.info('Bed sorted')
    ## Calculate coverage and scale to CPM
    cov = 1e6 / BedTool(f_sort).count()
    f_bg = f'{f_sort.split(".")[0]}.bdg'
    f_genome = os.path.join(chromograph.__path__[0], 'references/male.GRCh38.chrom.sizes')
    os.system(f'bedtools genomecov -i {f_sort} -g {f_genome} -scale {cov} -bg > {f_bg}')
    
    if verbose:
        logging.info('Bedgraph created')
    ## Convert to bigwig
    outfile = os.path.join(peak_dir, f'cluster_{str(cluster)}.bw')
    pybedtools.contrib.bigwig.bedgraph_to_bigwig(BedTool(f_bg), genome='hg38', output=outfile)
    
    ## Clean up
    os.system(f'rm {f_bg} {f_sort}')
    if verbose:
        logging.info(f'finished {cluster}')
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

def generate_peak_matrix_dict(id, cells, sample_dir, peak_dir, annot, verbose=True):
    '''
    '''

    ## Check if loom already exists
    loom_file = os.path.join(peak_dir, f'{id}_peaks.loom')

    if not os.path.exists(loom_file):
        try:
            dict_file = os.path.join(peak_dir, f'{id}.pkl')

            if not os.path.exists(dict_file):
                Count_peaks(id, cells, sample_dir, peak_dir, os.path.join(peak_dir, 'Compounded_peaks.bed'), )
            
            if verbose:
                logging.info("Generating Sparse matrix")
            col = []
            row = []
            v = []
            cix = 0
            IDs = []

            # Order dict for rows
            r_dict = {k: v for v,k in enumerate(annot['ID'])}

            ## Generate sparse peak lists
            Counts = pkl.load(open(dict_file, 'rb'))
            for cell in Counts:
                if len(Counts[cell]) > 0:
                    for key in (Counts[cell]):
                        col.append(cix)
                        row.append(r_dict[key])
                        v.append(np.int8(Counts[cell][key]))
                cix+=1
                IDs.append(cell)

            ## Convert to sparse matrix
            matrix = sparse.coo_matrix((np.ones(len(row)), (row,col)), shape=(len(r_dict.keys()), len(IDs))).tocsc()
            counts = sparse.coo_matrix((np.nan_to_num(v), (row,col)), shape=(len(r_dict.keys()), len(IDs))).tocsc()
            if verbose:
                logging.info(f'Matrix has shape {matrix.shape} with {matrix.nnz} elements')
                logging.info(f'Generating temporary loom file')

            ## Create loomfile
            if verbose:
                logging.info("Constructing loomfile")

            loompy.create(filename=loom_file, 
                        layers={'':matrix, 'Counts': counts}, 
                        row_attrs=annot, 
                        col_attrs={'CellID': np.array(IDs)})
            
            ## Remove pkl
            os.system(f'rm {dict_file}') 

            return
        except Exception as e:
            logging.info(f'Error in sample: {id}')
            logging.info(e)
    return

def generate_peak_matrix(id, cells, sample_dir, peak_dir, annot, verbose=True):
    '''
    '''

    ## Check if loom already exists
    loom_file = os.path.join(peak_dir, f'{id}_peaks.loom')

    if not os.path.exists(loom_file):
        try:
            
            counts = Count_peaks_matrix(id, cells, sample_dir, peak_dir, os.path.join(peak_dir, 'Compounded_peaks.bed'))
            counts.data = np.nan_to_num(counts.data, copy=False)

            ## Create binary matrix
            matrix = sparse.csc_matrix((np.ones(counts.nnz), counts.nonzero()), shape=counts.shape)

            if verbose:
                logging.info(f'Matrix has shape {matrix.shape} with {matrix.nnz} elements')
                logging.info(f'Generating temporary loom file')

            loompy.create(filename=loom_file, 
                        layers={'':matrix, 'Counts': counts}, 
                        row_attrs=annot, 
                        col_attrs={'CellID': cells})

            return
        except Exception as e:
            logging.info(f'Error in sample: {id}')
            logging.info(e)
    return

def merge_fragments(chunk, peakdir):
    '''
    '''
    files = np.array(chunk[1])
    fmerge = os.path.join(peakdir, f'cluster_{chunk[0]}.tsv.gz')
    missing = 0

    with open(fmerge, 'wb') as out:
        for f in files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    shutil.copyfileobj(file, out)
            else:
                f2 = '/' + os.path.join(*f.split('/')[:-1], f"{f.split('/')[-1].split('.')[0]}-1.tsv.gz")
                if os.path.exists(f2):
                    with open(f2, 'rb') as file:
                        shutil.copyfileobj(file, out)
                else:
                    missing += 1
    if missing > 0:
        logging.info(f'Finished with cluster {chunk[0]}, {missing} missing cells')
    return