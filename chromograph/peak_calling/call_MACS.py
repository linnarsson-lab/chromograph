import os
import sys
import logging
import pybedtools
import MACS2
import tempfile
import numpy as np
import shutil

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

    ## Handle tmpdir size
    tmpdir = tempfile.mkdtemp(dir = os.getcwd())

    ## Call Peaks
    # cmd = f'{macs_path} callpeak -t {fbed} -f BEDPE -g hs --nomodel --shift 100 --ext 200 --qval 5e-2 -B --SPMR -n {fpeaks}'
    cmd = f'{macs_path} callpeak -t {fbed} -f BEDPE -g hs --nomodel --shift 37 --ext 73 --qval 1e-2 -B --SPMR -n {fpeaks} --tempdir {tmpdir}'  ## ENCODE standard
    os.system(cmd)
    
    ## We only need the narrowPeak file, so clean up the rest
    # os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_peaks.xls')))
    # os.system("rm {}".format(os.path.join(pf, 'cluster_' + str(clus) + '_summits.bed')))
    os.system(f"rm {os.path.join(pf, 'cluster_' + str(clus) + '_treat_pileup.bdg')}")  ## Convert this track to BigWig
    os.system(f"rm {os.path.join(pf, 'cluster_' + str(clus) + '_control_lambda.bdg')}")
    shutil.rmtree(tmpdir)  ## Convert this track to BigWig

    return f'Cluster {clus} completed'