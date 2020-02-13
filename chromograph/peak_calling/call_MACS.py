import os
import sys
import logging
from pybedtools import BedTool
import MACS2

sys.path.append('/home/camiel/chromograph/')
from chromograph.peak_calling.utils import *

def call_MACS(data, pf, macs_path):
    """
    Call peaks based on BED aggregate files.

    args:
        data        list of [clustername, path to BED-file]
        pf          path to peaks directory
        macs_path   path to MACS

    return:
        location of aggregated peaks file
    """
    clus = data[0]
    fbed = data[1]
    fpeaks = os.path.join(pf, "cluster_{}".format(clus))

    ## Call Peaks
    cmd = "{} callpeak -t {} -f BEDPE -g hs --nomodel --shift 100 --ext 200 --qval 5e-2 -B --SPMR -n {}".format(macs_path, fbed, fpeaks)
    os.system(cmd)

    logging.info('Called peaks for cluster {} out of {}'.format(clus, np.unique(ds.ca['Clusters'])))
    
    ## We only need the narrowPeak file, so clean up the rest
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_peaks.xls')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_control_lambda.bdg')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_summits.bed')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_treat_pileup.bdg')))
 
    return "Cluster {} completed".format(clus)