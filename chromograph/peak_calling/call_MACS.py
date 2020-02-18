import os
import sys
import logging
from pybedtools import BedTool
import MACS2
import numpy as np

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
    cmd = f'{macs_path} callpeak -t {fbed} -f BEDPE -g hs --nomodel --shift 100 --ext 200 --qval 5e-2 -B --SPMR -n {fpeaks}'
    os.system(cmd)

    logging.info(f'Called peaks for cluster {clus}')
    
    ## We only need the narrowPeak file, so clean up the rest
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_peaks.xls')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_control_lambda.bdg')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_summits.bed')))
    os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_treat_pileup.bdg')))
 
    return f'Cluster {clus} completed'