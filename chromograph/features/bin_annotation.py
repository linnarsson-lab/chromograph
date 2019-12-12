import os
import numpy as np
import loompy

from pybedtools import BedTool

def Bin_annotation(ds: loompy.LoomConnection, ref) -> None:
    """
    Annotate bins with the closest protein coding gene. Adds Gene symbols, ensemble IDs 
    and distance to gene as row attributes
    
    Args:
        ds                    loompy connected file
        ref                   path to folder containing annotation genome. Must contain 'genes/genes.gtf' 

    Remarks:
        Currently annotating based only on the closest protein coding gene
    """
    logging.info(f"Loading genes reference set")
    genes = BedTool(os.path.join(ref, 'genes', 'genes.gtf'))

    coding = []
    for x in genes:
        if np.logical_and(x['gene_type'] == 'protein_coding', x[2] == 'gene'):
            coding.append(x)
    coding = BedTool(coding)

    bins = np.stack([ds.ra['chrom'], ds.ra['start'], ds.ra['end'], ds.ra['ID']]).T.tolist()
    bins = BedTool(bins)

    ## To simplify analysis we only annotate bins with the first hit if multiple genes are at the same distance
    logging.info(f"Overlapping genes with bins")
    annot = bins.sort().closest(coding.sort(), d=True, t='first')
    
    an_dict = {'Accession': [], 'Gene': [], 'Distance': [], 'ID': []}
    for x in annot:
        feats = x[12].split(';')[:-1]
        feats = [x.split(' ') for x in feats]
        feats = {k:v for k,v in feats}
        an_dict['Accession'].append(feats['gene_id'].strip('"'))
        an_dict['Gene'].append(feats['gene_name'].strip('"'))
        an_dict['Distance'].append(int(x[13]))
        an_dict['ID'].append(int(x[3]))
    an_dict = {x: np.array(an_dict[x]) for x in an_dict}
    an_dict = {x: an_dict[x][an_dict['ID'].argsort()] for x in an_dict}

    ## Adding annotation to loomfile
    for x in ['Accession', 'Gene', 'Distance']:
        ds.ra[x] = an_dict[x]