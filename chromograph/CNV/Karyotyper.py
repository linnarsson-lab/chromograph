import numpy as np
import os, sys
import loompy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from statsmodels.nonparametric.smoothers_lowess import lowess
from cytograph.pipeline import Tempname

import chromograph
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

def calc_cpu(n_cells):
    n = np.array([1e2, 1e3, 1e4, 1e5, 5e5, 1e6, 2e6])
    cpus = [1, 3, 7, 14, 28, 28, 56]
    idx = (np.abs(n - n_cells)).argmin()
    return cpus[idx]

class Karyotyper:
    def __init__(self, max_rv = 5, min_fraction:float = .05, window_size: int = 100, smoothing_bandwidth: int = 0, marker: str = 'PTPRC'):
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
            
            
        sig_std = np.percentile(np.std(self.yratio[valid_clusters], axis=1), 99)
        aneuploid = np.std(self.yratio, axis=1) > sig_std
        valid = np.isin(ds.ca.Clusters, np.where(aneuploid)[0])
        ds.ca.Aneuploid = valid
        
        return
    
    def plot(self, ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, markers: list = ['PTPRC', 'SOX10', 'EGFR', 'AQP4', 'DCN'], reference_marker: str = 'PTPRC', out_file: str = None):
        '''
        '''
        logging.info(f'Generating plot')
        fig, ax = plt.subplots(2,4, figsize=(16,8))
        ax = ax.flatten()

        XY = ds.ca[self.config.params.main_emb]

        for i, marker in enumerate(markers):

            markerpeaks = np.where(dsagg.ra['Gene Name'] == marker)[0]
            X = np.sum(ds[markerpeaks,:], axis=0)

            ax[i].scatter(XY[:,0], XY[:,1], c='lightgray', s=1)
            ax[i].scatter(XY[X>0,0], XY[X>0,1], c=X[X>0], cmap = 'YlOrRd', s=1)
            ax[i].set_title(marker)

        names, labels = np.unique(ds.ca.Clusters, return_inverse=True)
        colors = colorize(names)
        ax[i+1].scatter(XY[:,0], XY[:,1], c=colors[labels], s=.5)
        ax[i+1].set_title(f'{ds.shape[1]} Cells')

        markerpeaks = np.where(dsagg.ra['Gene Name'] == reference_marker)[0]
        X = np.mean(dsagg['residuals'][markerpeaks,:], axis=0)
        valid_clusters = np.where(X > 3)[0]
        valid = np.isin(ds.ca.Clusters, valid_clusters)

        ax[i+2].scatter(XY[:,0], XY[:,1], c='lightgray', s=1)
        ax[i+2].scatter(XY[valid,0], XY[valid,1], c='red', s=1)
        ax[i+2].set_title(f'Reference clusters')

        valid = np.where(ds.ca.Aneuploid)[0]
        ax[i+3].scatter(XY[:,0], XY[:,1], c='lightgray', s=1)
        ax[i+3].scatter(XY[valid,0], XY[valid,1], c='blue', s=1)
        ax[i+3].set_title(f'Aneuploidity (P<0.01)')

        for current_ax in ax:
            current_ax.axis('off')
            
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
            if len(pos)>0:
                midpoint = int((pos[-1]+pos[0])/2)
                xpoint.append(midpoint)
                labels.append(k.strip('chr'))
        ax2.set_xticks(ticks=xpoint)
        ax2.set_xticklabels(labels=np.array(labels))
        ax2.set_aspect('auto')

        if out_file:
            fig.savefig(out_file)
        else:
            name = ds.filename.split('/')[-2]
            fig.savefig(os.path.join(self.config.paths.build, name, 'exported', 'Karyotype.png'), dpi=300)
            fig2.savefig(os.path.join(self.config.paths.build, name, 'exported', 'Karyotype_heatmap.png'), dpi=300)

    def generate_punchcards(self, config, ds: loompy.LoomConnection, dsagg: loompy.LoomConnection, python_exe=None):
        ds.ca.Split = ds.ca.Aneuploid
        loom_file = ds.filename
        subset = loom_file.split('/')[-1].split('_')[0]

        out_dir = os.path.join(config.paths.build, subset, "exported", 'split')

        with Tempname(out_dir) as exportdir:
            os.mkdir(exportdir)

            # Calculate split sizes
            clusters = ds.ca.Aneuploid
            sizes = np.bincount(clusters)
            logging.info("Creating punchcard")
            with open(f'{config.paths.build}/punchcards/{subset}.yaml', 'w') as f:
                for i in np.unique(clusters):
                    # Calc cpu and memory
                    n_cpus = calc_cpu(sizes[i])
                    memory = 750 if n_cpus == 56 else config.execution.memory
                    # Write to punchcard
                    name = 'Aneuploid' if i else 'Euploid'
                    f.write(f'{name}:\n')
                    f.write('  include: []\n')
                    f.write(f'  onlyif: Aneuploid == {i}\n')
                    f.write('  execution:\n')
                    f.write(f'    n_cpus: {n_cpus}\n')
                    f.write(f'    memory: {memory}\n')
                    
            # create submit file for split
            workflow = chromograph.__path__[0] + '/pipeline/subset_workflow.py'
            exdir = os.path.join(config.paths.build, 'submits', 'split')
            logdir = os.path.join(config.paths.build, 'logs')
            if not os.path.isdir(exdir):
                os.mkdir(exdir)
            if not python_exe:
                python_exe = os.path.abspath(sys.executable)

            n_cpus = max([calc_cpu(sizes[i]) for i in np.unique(clusters)])
            file = os.path.join(exdir, f"Split_{subset}.condor")
            with open(file, "w") as f:
                names = []
                for i in np.unique(clusters):
                    name = 'Aneuploid' if i else 'Euploid'
                    names.append(subset + '_' + name)
                
                delim = '\n'
                f.write(f"""
                        getenv       = true
                        environment  = "PYTHONPATH=$ENV(CONDA_PREFIX)/lib/python3.7 PYTHONHOME=$ENV(CONDA_PREFIX)"
                        executable   = {python_exe}
                        arguments    = "{workflow} $(set)"
                        log          = {logdir}/$(set).log
                        output       = {logdir}/$(set).out
                        error        = {logdir}/$(set).error
                        request_cpus = {n_cpus}
                        queue set in (
                        {delim.join(names)}
                        )\n
                        """)
        return