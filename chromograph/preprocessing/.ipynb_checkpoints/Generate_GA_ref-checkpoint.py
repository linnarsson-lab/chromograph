
import numpy as np
import os
import sys
import gzip
import pybedtools
from pybedtools import BedTool

sys.path.append('/home/camiel/chromograph/')
import chromograph

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

ref_assembly = sys.argv[1]
ref = sys.argv[2]

logging.info(f'Generating Gene-Accessibility reference from {ref_assembly}')

genes = BedTool(os.path.join(ref, 'genes', 'genes.gtf'))

gb = []
for x in genes:
    if np.logical_and(x[2] == 'gene', x.attrs['gene_type'] == 'protein_coding'):
        gb.append(x)
        
logging.info(f'Filtered out non-protein coding genes')
        
gb = BedTool(gb)
print(len(gb))

if ref_assembly == 'GRCh38':
    gb = gb.slop(s=True, l=pad, r=0, genome = 'hg38')
else:
    gb = gb.slop(s=True, l=pad, r=0, genome = ref_assembly)
    
f = os.path.join(chromograph.__path__[0], f'references/{ref_assembly}_genes_{int(pad/1000)}kbprom.bed')
gb.saveas(f)