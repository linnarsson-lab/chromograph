import os, glob
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from pybedtools import BedTool
from typing import List
import numpy as np
import pyBigWig
from cytograph.visualization.colors import colorize

def plot_bigWig_coords_rows(pos_file, peak_dir, out_file, IDs: List[str] = None, padding = 300, dpi=600, colors:dict=None, N_rows = None):
    '''
    '''
    if IDs is None:
        ## Plot track for all clusters
        files = glob.glob(os.path.join(peak_dir, '*.bw'))
        IDs = [f.split('/')[-1].split('.')[0] for f in files]
        idx = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(IDs))]
        files = [files[i] for i in idx]
    else:
        ## Generate paths to requested clusters
        files = [os.path.join(peak_dir, f'{i}.bw') for i in IDs]

    if colors == None:
        c = colorize(np.arange(len(IDs)))
        colors = {k:v for k, v in zip(IDs, c)}

    pos = np.array(BedTool(pos_file))
    if N_rows:
        pos = pos[:N_rows]
    chrom = pos[:,0]
    start = pos[:,1].astype(int) - padding
    end = pos[:,2].astype(int) + padding
        
    width = len(files)
    heigth = pos.shape[0]
    fig, ax = plt.subplots(pos.shape[0],len(files), figsize=(width, heigth))
      
    ## Plot each track
    for r in range(pos.shape[0]):
        data = []
        for c, file in enumerate(files):
            with pyBigWig.open(file) as bw:
                data.append(np.nan_to_num(bw.values(chrom[r], start[r], end[r], numpy=True),0))

        height = max([max(x) for x in data])
        x = np.fromiter(range(len(data[0])), dtype=int)
        x2 = x[padding:-padding]
        for i, v in enumerate(data):
            col = colors[IDs[i]]
            ax[r,i].plot(v, color = col, lw=.5)
            ax[r,i].fill_between(x,v, alpha=.5, color = col)
            ax[r,i].fill_between(x2,v[x2], alpha=.7, color = col)
            ax[r,i].set_ylim(0,height)
            ax[r,i].axis('off')
        plt.tight_layout()
        
        if r == 0:
            for c in range(len(data)):
                ax[0,c].set_title(IDs[c])
            
        
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight')

def plot_bigWig_coords_cols(pos_file, bigwig_dir, out_file, IDs: List[str] = None, padding = 300, dpi=600, colors:dict=None, N_cols = None, plot_rownames: bool = True):
    '''
    '''
    if IDs is None:
        ## Plot track for all clusters
        files = glob.glob(os.path.join(bigwig_dir, '*.bw'))
        IDs = [f.split('/')[-1].split('.')[0] for f in files]
        idx = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(IDs))]
        files = [files[i] for i in idx]
    else:
        ## Generate paths to requested clusters
        files = [os.path.join(bigwig_dir, f'{i}.bw') for i in IDs]

    if colors == None:
        c = colorize(np.arange(len(IDs)))
        colors = {k:v for k, v in zip(IDs, c)}

    pos = np.array(BedTool(pos_file))
    if N_cols:
        pos = pos[:N_cols]
    chrom = pos[:,0]
    start = pos[:,1].astype(int) - padding
    end = pos[:,2].astype(int) + padding
        
    width = pos.shape[0]
    heigth = len(files)
    fig, ax = plt.subplots(len(files),pos.shape[0], figsize=(width, heigth))
      
    ## Plot each track
    for c in range(pos.shape[0]):
        data = []
        for r, file in enumerate(files):
            with pyBigWig.open(file) as bw:
                data.append(np.nan_to_num(bw.values(chrom[c], start[c], end[c], numpy=True),0))

        width = max([max(x) for x in data])
        x = np.fromiter(range(len(data[0])), dtype=int)
        x2 = x[padding:-padding]
        for i, v in enumerate(data):
            row = colors[IDs[i]]
            ax[i,c].plot(v, color = row, lw=.5)
            ax[i,c].fill_between(x,v, alpha=.5, color = row)
            ax[i,c].fill_between(x2,v[x2], alpha=.7, color = row)
            ax[i,c].set_ylim(0,width)
            ax[i,c].axis('off')
        plt.tight_layout()
        
        if plot_rownames:
            if c == 0:
                for r in range(len(data)):
                    ax[r,0].set_title(IDs[r])
            
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight')