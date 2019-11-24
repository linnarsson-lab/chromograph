import os
import logging
from pybedtools import BedTool
import MACS2

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *

def call_MACS(data, pf):

    clus = data[0]
    fragments = data[1]
    frags = [strFrags_to_list(x) for x in fragments]
    frags = [x for l in frags for x in l]
    logging.info("Total fragments in cluster {}:  {}".format(clus, len(frags)))

    fbed = os.path.join(pf, "fragments_cluster_{}.bed.gz".format(clus))
    fpeaks = os.path.join(pf, "cluster_{}".format(clus))

    logging.info("{},  {}".format(g, fpeaks))
    bed = BedTool(frags)
    bed.saveas(fbed)

    ## Call Peaks
    cmd = "{} callpeak -t {} -f BEDPE -g {} --nomodel --shift 100 --ext 200 --qval 5e-2 -B --SPMR -n {}".format(macs_path, fbed, g, fpeaks)
    os.system(cmd)

    logging.info('Called peaks for cluster {} out of {}'.format(clus, np.unique(ds.ca['Clusters'])))
    
    ## We only need the narrowPeak file, so clean up the rest
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_peaks.xls')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_control_lambda.bdg')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_summits.bed')))
 
    return "Cluster {} completed".format(clus)