## QC_plots
import matplotlib.pyplot as plt
import numpy  as np
from pybedtools import BedTool
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
import logging

def sample_QC_plots(ds, fragments, outfile):
    fig, ax = plt.subplots(2,2, figsize=(12,10))
    XY = ds.ca.TSNE
    
    im = ax[0,0].scatter(XY[:,0], XY[:,1], marker ='.', lw=0, c=np.log2(ds.ca.passed_filters))
    cbar = fig.colorbar(im, ax=ax[0,0], orientation='vertical', shrink=.5, label='log2(fragments)');
    ax[0,0].set_title('Fragment Count')
    ax[0,0].axis('off')

    im = ax[0,1].scatter(XY[:,0], XY[:,1], marker ='.', lw=0, c=ds.ca.FRtss, cmap='gnuplot', vmin=0)
    cbar = fig.colorbar(im, ax=ax[0,1], orientation='vertical', shrink=.5, label='fraction  TSS');
    ax[0,1].set_title('TSS overlap')
    ax[0,1].axis('off')
    
    im = ax[1,0].scatter(XY[:,0], XY[:,1], marker ='.', lw=0, c=ds.ca.DoubletFinderScore, cmap='jet', vmin=0)
    cbar = fig.colorbar(im, ax=ax[1,0], orientation='vertical', shrink=.5, label='Doublet Score');
    ax[1,0].set_title('Doublet Score')
    ax[1,0].axis('off')
    
    x = 5e7
    l = []
    bd = BedTool(fragments)
    for i, row in tqdm(enumerate(bd)):
        l.append(int(row[2])-int(row[1]))
        if i >=x:
            break
    ax[1,1].hist(l, bins=100, alpha=.5, range=(0,1000), density=True);
    ax[1,1].yaxis.set_major_formatter(PercentFormatter(1));
    ax[1,1].set_title('Fragment size distribution')
    ax[1,1].set_ylabel('Fragments (%)')
    ax[1,1].set_xlabel('Fragment size (bp)')
    
    plt.savefig(outfile, dpi=300)