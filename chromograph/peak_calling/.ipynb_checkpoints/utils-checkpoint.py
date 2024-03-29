import pybedtools
from pybedtools import BedTool

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
    with open(file) as f:
        i = 0
        for line in f:
            if i == 0:
                cols = ['ID'] + line.split('\t')[1:]
                cols = [x.rstrip() for x in cols]
            if i > 0 :
                table.append([x.rstrip() for x in line.split('\t')])
            i += 1

    table = np.array(table)
    return table

def reorder_by_IDs(mat, IDs):
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

    return np.array(ntable)

def Count_peaks(cells, sample_dir, f_peaks, q):
    '''
    Count peaks
    '''
    Count_dict = {k: {} for k in cells}
    peaks = BedTool(f_peaks)
    i = 0
    for x in cells:
        try:
            s, c = x.split(':')
            f = os.path.join(sample_dir, s, 'fragments', f'{c}.tsv.gz')
            cBed = BedTool(f)
            pks = peaks.intersect(cBed, wa=True)

            cDict = {}
            for line in pks:
                cDict[line[3]] = 1

            Count_dict[x] = cDict
            i += 1
            if i%1000==0:
                logging.info(f'Finished counting {i} cells')
        except:
            Count_dict[x] = []
    logging.info('Finished job')
    return q.put(Count_dict)

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