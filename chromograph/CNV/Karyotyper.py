import numpy as np
import os
import loompy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from statsmodels.nonparametric.smoothers_lowess import lowess

from chromograph.pipeline.utils import div0
from chromograph.pipeline import config
from cytograph.plotting.colors import colorize
import logging

## Setup logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')
logger = logging.getLogger()

class Karyotyper:
    def __init__(self, max_rv = 10, min_fraction:float = .01, window_size: int = 100, smoothing_bandwidth: int = 0, marker: str = 'PTPRC'):
        self.config = config.load_config()
        self.max_rv = max_rv
        self.min_fraction = min_fraction
        self.window_size = window_size
        self.smoothing_bandwidth = smoothing_bandwidth
        self.marker = marker
        
        self.chr_starts = {
            'chr1': 0,
            'chr2': 248956422,
            'chr3': 491149951,
            'chr4': 689445510,
            'chr5': 879660065,
            'chr6': 1061198324,
            'chr7': 1232004303,
            'chr8': 1391350276,
            'chr9': 1536488912,
            'chr10': 1674883629,
            'chr11': 1808681051,
            'chr12': 1943767673,
            'chr13': 2077042982,
            'chr14': 2191407310,
            'chr15': 2298451028,
            'chr16': 2400442217,
            'chr17': 2490780562,
            'chr18': 2574038003,
            'chr19': 2654411288,
            'chr20': 2713028904,
            'chr21': 2777473071,
            'chr22': 2824183054,
            'chrX': 2875001522,
            'chrY': 3031042417
        }
        
    def windowed_mean(self, x: np.ndarray, n: int):
        if len(x) == 0:
            return x
        y = np.zeros_like(x)
        for ix in range(len(x)):
            i = min(ix, len(x) - n)
            w = x[i:i + n]
            y[ix] = np.mean(w)
        return y
        
    def fit(self, ds: loompy.LoomConnection, dsagg: loompy.LoomConnection):
        '''
        Fit the karyotyper and return the labels
        '''
        logging.info(f'Fitting Karyotyper')
        
        ## Identify housekeeping peaks
        freq = dsagg.ra.totals / ds.shape[1]        
        X = dsagg['residuals'][:,:]
        variance = X.var(axis=1)
        means = X.mean(axis=1)
        housekeeping = (variance < self.max_rv) & (freq > self.min_fraction)
        logging.info(f'Number of housekeeping peaks: {np.sum(housekeeping)}')
        
        # Order by genomic position
        chrs = dsagg.ra.Chr[housekeeping]
        starts = dsagg.ra.Start[housekeeping].astype(int)
        for chrom in self.chr_starts.keys():
            starts[chrs == chrom] += self.chr_starts[chrom]
        ordering = np.argsort(starts)
        self.chromosomes = chrs[ordering]
        self.starts = starts[ordering]
        
        ## Get the CPM of housekeeping peaks
        y_sample = dsagg['CPM'][:,:][housekeeping,:]
        y_sample = y_sample[ordering,:]
        y_sample_mean = y_sample.mean(0)
        
        ## Identify reference cells by marker gene enrichment
        markerpeaks = np.where(dsagg.ra['Gene Name'] == self.marker)[0]
        X = np.mean(dsagg['residuals'][markerpeaks,:], axis=0)
        valid_clusters = np.where(X > 3)[0]
        logging.info(f'Reference clusters: {dsagg.ca.Clusters[valid_clusters]}')
        
        ## Get reference track
        y_ref = dsagg['CPM'][:,valid_clusters][housekeeping,:]
        y_ref = y_ref[ordering,:].mean(axis=1)
        y_ref_mean = y_ref.mean()
        
        ## Bin locally along chromosome
        logging.info(f'Binning')
        for chrom in self.chr_starts:
            selected = self.chromosomes == chrom
            if selected.sum() == 0:
                continue
            y_ref[selected] = self.windowed_mean(y_ref[selected], self.window_size)
            for i in range(y_sample.shape[1]):
                y_sample[selected, i] = self.windowed_mean(y_sample[selected,i], self.window_size)
        
        ## Center around reference residuals
        logging.info(f'Centering')
        self.yratio = (div0(y_sample.T, y_ref).T * y_ref_mean / y_sample_mean).T
        
        if self.smoothing_bandwidth > 0:
            logging.info(f'Smoothing')
            # Loess smoothing along each chromosome
            yratio_smooth = np.copy(self.yratio)
            for chrom in self.chr_starts.keys():
                selected = (self.chromosomes == chrom)
                for i in range(yratio_smooth.shape[0]):
                    yratio_smooth[i, selected] = lowess(yratio_smooth[i, selected], self.starts[selected], frac=min(0.5, self.smoothing_bandwidth / selected.sum()), return_sorted=False)
        else:
            yratio_smooth = None
            
            
        sig_std = np.percentile(np.std(self.yratio[valid_clusters], axis=1), 95)
        aneuploid = np.std(self.yratio, axis=1) > sig_std
        valid = np.isin(ds.ca.Clusters, np.where(aneuploid)[0])
        ds.ca.Aneuploid = valid
        
        return
    
    def plot(self, ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, markers: list = ['PTPRC', 'SOX10', 'EGFR', 'AQP4', 'DCN'], reference_marker: str = 'PTPRC'):
        '''
        '''
        logging.info(f'Generating plot')
        fig, ax = plt.subplots(2,4, figsize=(16,8))
        ax = ax.flatten()

        for i, marker in enumerate(markers):

            markerpeaks = np.where(dsagg.ra['Gene Name'] == marker)[0]

            X = np.sum(ds[markerpeaks,:], axis=0)

            ax[i].scatter(ds.ca.TSNE[:,0], ds.ca.TSNE[:,1], c='lightgray', s=1)
            ax[i].scatter(ds.ca.TSNE[X>0,0], ds.ca.TSNE[X>0,1], c=X[X>0], cmap = 'viridis', s=.5)
            ax[i].set_title(marker)

        names, labels = np.unique(ds.ca.Clusters, return_inverse=True)
        colors = colorize(names)
        ax[i+1].scatter(ds.ca.TSNE[:,0], ds.ca.TSNE[:,1], c=colors[labels], s=.5)
        ax[i+1].set_title(f'{ds.shape[1]} Cells')

        markerpeaks = np.where(dsagg.ra['Gene Name'] == reference_marker)[0]
        X = np.mean(dsagg['residuals'][markerpeaks,:], axis=0)
        valid_clusters = np.where(X > 3)[0]
        valid = np.isin(ds.ca.Clusters, valid_clusters)

        ax[i+2].scatter(ds.ca.TSNE[:,0], ds.ca.TSNE[:,1], c='lightgray', s=1)
        ax[i+2].scatter(ds.ca.TSNE[valid,0], ds.ca.TSNE[valid,1], c='red', s=1)
        ax[i+2].set_title(f'Reference clusters')

        valid = np.where(ds.ca.Aneuploid)[0]
        ax[i+3].scatter(ds.ca.TSNE[:,0], ds.ca.TSNE[:,1], c='lightgray', s=1)
        ax[i+3].scatter(ds.ca.TSNE[valid,0], ds.ca.TSNE[valid,1], c='blue', s=1)
        ax[i+3].set_title(f'Aneuploidity (P<0.05)')

        bottom = plt.cm.get_cmap('Oranges', 128)
        middle = np.full((50, 4), 0.99)
        middle[:, 3] = 1
        top = plt.cm.get_cmap('Blues_r', 128)

        newcolors = np.vstack((top(np.linspace(0, 1, 128)), middle, bottom(np.linspace(0, 1, 128))))
        cmp = ListedColormap(newcolors, name='OrangeBlue')

        fig2, ax2 = plt.subplots(figsize=(16,6))
        ax2.set_title(f'Karyotype by cluster')
        im = ax2.imshow(self.yratio, cmap = cmp, vmin=0, vmax=2)
        fig2.colorbar(im, ax=ax2, orientation='vertical', shrink=.5)
        for k in self.chr_starts:
            x = np.where(abs(self.starts - self.chr_starts[k])== np.min(abs(self.starts - self.chr_starts[k])))[0]
            ax2.axvline(x, c='black', lw=1)

        labels, xpoint = [],[]
        for k in self.chr_starts:
            pos = np.where(self.chromosomes==k)[0]
            midpoint = int((pos[-1]+pos[0])/2)
            xpoint.append(midpoint)
            labels.append(k.strip('chr'))
        ax2.set_xticks(ticks=xpoint)
        ax2.set_xticklabels(labels=np.array(labels))
        ax2.set_aspect('auto')

        name = ds.filename.split('/')[-2]
        plt.savefig(os.path.join(self.config.paths.build, name, 'exported', 'Karyotype.png'))