from collections import defaultdict
import os
from tqdm import tqdm

from cytograph.pipeline.punchcards import (Punchcard, PunchcardDeck, PunchcardSubset,PunchcardView)
from cytograph.manifold import BalancedKNN
from cytograph.plotting import manifold
from cytograph.embedding import art_of_tsne

from chromograph.pipeline import config
from chromograph.pipeline.utils import transfer_ca, div0
from chromograph.pipeline.TF_IDF import TF_IDF
from chromograph.pipeline.PCA import PCA
from chromograph.peak_analysis.feature_selection_by_PearsonResiduals import FeatureSelectionByPearsonResiduals
from chromograph.peak_analysis.Peak_Aggregator import Peak_Aggregator
from chromograph.RNA.RNA_analysis import RNA_analysis
from chromograph.plotting.QC_plot import QC_plot
from chromograph.motifs.motif_aggregation import motif_aggregator
from chromograph.peak_calling.utils import export_bigwig

import numpy as np
from scipy import sparse
import loompy
import multiprocessing as mp

from umap import UMAP

import logging
logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

class Pool_Splits:
    """
    Workflow for the final pooling step, which collects its cells from all the leaf subsets
    """
    def __init__(self, deck: PunchcardDeck, config) -> None:
        self.deck = deck
        self.config = config
        self.main_file = os.path.join(self.config.paths.build, 'All', 'All_peaks.loom')
        self.outdir = os.path.join(self.config.paths.build, 'Pool', 'exported')
        self.peak_file = os.path.join(self.config.paths.build, 'Pool', 'Pool_peaks.loom')
        self.peak_agg = os.path.join(self.config.paths.build, 'Pool', 'Pool_peaks.agg.loom')
        
        self.depth_key = 'NPeaks'

        # Merge pool-specific config
        if os.path.exists(os.path.join(self.config.paths.build, "pool_config.yaml")):
            self.config.merge_with(os.path.join(self.config.paths.build, "pool_config.yaml"))

    def collect_cells(self) -> None:
        clusters: List[int] = []
        next_cluster = 0

        # Check that all the inputs exist
        logging.info(f"Checking that all input files are present")
        err = False
        for subset in self.deck.get_leaves():
            if not os.path.exists(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_peaks.loom")):
                logging.error(f"Punchcard file '{subset.longname()}.loom' is missing")
                err = True
        if err:
            sys.exit(1)

        if not os.path.isdir(os.path.join(self.config.paths.build, 'Pool')):
            os.mkdir(os.path.join(self.config.paths.build, 'Pool'))
            
        type_dict = defaultdict(list)
        shape_dict = defaultdict(list)
        splits = []
        files = []
        selections = []
        for subset in self.deck.get_leaves():
            logging.info(f"Collecting metadata from {subset.longname()}")
            files.append(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_peaks.loom"))
            with loompy.connect(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_peaks.loom"), mode="r") as ds:
                if 'ClusterName' in ds.ca:
                    valid = np.where(ds.ca.ClusterName != 'unclear')[0]
                    clusters.append(list(ds.ca.Clusters[valid] + next_cluster))
                    splits.append(np.repeat(subset.longname(), len(valid)))
                    selections.append(ds.ca.ClusterName != 'unclear')
                else:
                    clusters.append(list(ds.ca.Clusters + next_cluster))
                    splits.append(np.repeat(subset.longname(), ds.shape[1]))
                    selections.append(np.ones(ds.shape[1]).astype('bool'))
                next_cluster = max(clusters[-1]) + 1
                for k, v in ds.ca.items():
                    type_dict[k].append(v.dtype)
                    sh = 0 if len(v.shape) == 1 else v.shape[1]
                    shape_dict[k].append(sh)
        splits = np.array([x for s in splits for x in s])
        clusters = np.array([x for s in clusters for x in s])
        d = {k:v for v,k in enumerate(np.unique(clusters))}
        clusters = np.array([d[x] for x in clusters])
        skip_attrs = [k for k, v in type_dict.items() if not len(set(v)) == 1 or not len(set(shape_dict[k])) == 1]
        logging.info(f"Skipping attrs: {skip_attrs}")

        logging.info(f"Collecting all cells into {self.peak_file}")
        loompy.combine_faster(files, self.peak_file, selections=selections, key="ID", skip_attrs=skip_attrs)

        with loompy.connect(self.peak_file) as ds:
            ds.ca.SubClusters = ds.ca.Clusters
            ds.ca.Clusters = clusters
            ds.ca.Splits = splits
            logging.info(f"{ds.ca.Clusters.max() + 1} clusters")

            ## nonzero (nnz) counts per peak
            logging.info('Calculating peak and cell coverage')
            ds.ra['NCells'] = ds.map([np.count_nonzero], axis=0)[0]
            ds.ca['NPeaks'] = ds.map([np.count_nonzero], axis=1)[0]
            ds.ca['FRIP'] = div0(ds.ca.NPeaks, ds.ca.passed_filters)

            with loompy.connect(self.main_file) as dsa:
                match = {k:v for v, k in enumerate(dsa.ca.CellID)}
                TSNE_main, UMAP_main, UMAP3D_main = np.zeros((ds.shape[1],2)), np.zeros((ds.shape[1],2)), np.zeros((ds.shape[1],3))
                LSI_main = np.zeros((ds.shape[1], dsa.ca.LSI.shape[1]))
                clusters_main = np.zeros(ds.shape[1])
                for i, x in enumerate(ds.ca.CellID):
                    k = match[x]
                    TSNE_main[i] = dsa.ca['TSNE'][k]
                    UMAP_main[i] = dsa.ca['UMAP'][k]
                    UMAP3D_main[i] = dsa.ca['UMAP3D'][k]
                    LSI_main[i] = dsa.ca['LSI'][k]
                    clusters_main[i] = dsa.ca.Clusters[k]
                ds.ca.TSNE_main = TSNE_main
                ds.ca.UMAP_main = UMAP_main
                ds.ca.UMAP3D_main = UMAP3D_main
                ds.ca.LSI_main = LSI_main
                ds.ca.Clusters_main = clusters_main.astype(int)
                
            if not os.path.isdir(self.outdir):
                os.mkdir(self.outdir)

            ## Aggregate
            if 'Valid' in ds.ra:
                del ds.ra.Valid
            peak_agg = self.peak_file.split('.')[0] + '.agg.loom'
            peak_aggregator = Peak_Aggregator()
            peak_aggregator.fit(ds, peak_agg, motifs=False)

            ## Generate new embedding
            with loompy.connect(peak_agg) as dsagg:
                ds.ra.Valid, ds.ra.preCluster_residuals = dsagg.ra.Valid, dsagg.ra.residuals

            ## Faster LSI
            f_temp = ds.filename + '.tmp'
            if os.path.isfile(f_temp):
                os.remove(f_temp)
                
            logging.info(f'Making temp file')
            x = np.where(ds.ra.Valid)[0]
            loompy.create(f_temp, {'':ds.sparse(rows=x).tocsr().astype('float16')}, {'ID': ds.ra.ID[x]}, col_attrs=ds.ca)
            with loompy.connect(f_temp) as dst:
                dst.ra.Valid = np.ones(dst.shape[0])

                ## Term-Frequence Inverse-Data-Frequency ##
                logging.info(f'Performing TF-IDF')
                tf_idf = TF_IDF(layer='')
                tf_idf.fit(dst)
                dst.layers['TF-IDF'] = 'float16'
                progress = tqdm(total=dst.shape[1])
                for (_, selection, view) in dst.scan(axis=1, batch_size=self.config.params.batch_size):
                    dst['TF-IDF'][:,selection] = tf_idf.transform(view[:,:], selection)
                    progress.update(self.config.params.batch_size)
                progress.close()
                del tf_idf
                logging.info(f'Finished fitting TF-IDF')

                ## Fit PCA
                logging.info(f'Fitting PCA')
                pca = PCA(max_n_components = self.config.params.n_factors, layer= '', key_depth= self.depth_key, batch_keys = self.config.params.batch_keys)
                pca.fit(dst)

                ## Decompose data
                ds.ca.LSI = pca.transform(dst)

                logging.info(f'Finished PCA transformation')
                del pca

                ## Get correct embedding and metric
                decomp = ds.ca.LSI
            os.remove(f_temp)


            decomp = ds.ca.LSI
            ## Construct nearest-neighbor graph
            metric = 'euclidean' # jaccard js euclidean correlation cosine 
            logging.info(f"Computing balanced KNN (k = {self.config.params.k}) space using the '{metric}' metric")
            bnn = BalancedKNN(k=self.config.params.k, metric=metric, maxl=2 * self.config.params.k, sight_k=2 * self.config.params.k, n_jobs=-1)
            bnn.fit(decomp)
            knn = bnn.kneighbors_graph(mode='distance')
            knn.eliminate_zeros()
            mknn = knn.minimum(knn.transpose())
            # Convert distances to similarities
            logging.info(f'Converting distances to similarities knn shape.')
            max_d = knn.data.max()
            knn.data = (max_d - knn.data) / max_d
            mknn.data = (max_d - mknn.data) / max_d
            ds.col_graphs.KNN = knn
            ds.col_graphs.MKNN = mknn
            mknn = mknn.tocoo()
            mknn.setdiag(0)
            # Compute the effective resolution
            logging.info(f'Computing resolution')
            d = 1 - knn.data
            radius = np.percentile(d, 90)
            logging.info(f"  90th percentile radius: {radius:.02}")
            ds.attrs.radius = radius
            inside = mknn.data > 1 - radius
            rnn = sparse.coo_matrix((mknn.data[inside], (mknn.row[inside], mknn.col[inside])), shape=mknn.shape)
            ds.col_graphs.RNN = rnn

            ## Perform tSNE and UMAP
            logging.info(f"Computing 2D and 3D embeddings from latent space")
            logging.info(f"Art of tSNE with distance metrid: {metric}")
            ds.ca.TSNE = np.array(art_of_tsne(decomp, metric=metric))  # art_of_tsne returns a TSNEEmbedding, which can be cast to an ndarray (its actually just a subclass)
        
            logging.info(f'Generating UMAP from decomposition using metric {metric}')
            ds.ca.UMAP = UMAP(n_components=2, metric=metric, verbose=True).fit_transform(decomp)
            logging.info(f'Generating 3D UMAP from decomposition using metric {metric}')
            ds.ca.UMAP3D = UMAP(n_components=3, metric=metric, verbose=True).fit_transform(decomp)

            ## Plot manifold
            plot_attr = self.config.params.plot_attrs.copy()
            plot_attr.append('Splits')
            manifold(ds, os.path.join(self.outdir, f"Pool_peaks_UMAP.png"), embedding = 'UMAP')
            manifold(ds, os.path.join(self.outdir, f"Pool_peaks_TSNE.png"), embedding = 'TSNE')
            QC_plot(ds, os.path.join(self.outdir, f"Pool_peaks_UMAP_QC.png"), embedding = 'UMAP', attrs=plot_attr)
            QC_plot(ds, os.path.join(self.outdir, f"Pool_peaks_TSNE_QC.png"), embedding = 'TSNE', attrs=plot_attr)

        logging.info(f'Finished pooling peaks')

    def Pool_RNA(self) -> None:
        out_file = os.path.join(self.config.paths.build, 'Pool', 'Pool_RNA.loom')
        logging.info(f'Pooling RNA subsets')

        punchcards: List[str] = []
        clusters: List[int] = []
        punchcard_clusters: List[int] = []
        next_cluster = 0

        # Check that all the inputs exist
        logging.info(f"Checking that all input files are present")
        err = False
        for subset in self.deck.get_leaves():
            if not os.path.exists(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_RNA_imputed.loom")):
                logging.error(f"Punchcard file '{subset.longname()}.loom' is missing")
                err = True
        if err:
            sys.exit(1)
    
        type_dict = defaultdict(list)
        shape_dict = defaultdict(list)
        files = []
        selections = []
        with loompy.connect(self.peak_file) as ds_main:
            valid = set(ds_main.ca.CellID)
            for subset in self.deck.get_leaves():
                logging.info(f"Collecting metadata from {subset.longname()}")
                files.append(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_RNA_imputed.loom"))
                with loompy.connect(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_RNA_imputed.loom"), mode="r") as ds:
                    selections.append(np.array([x in valid for x in ds.ca.CellID]))
                    for k, v in ds.ca.items():
                        type_dict[k].append(v.dtype)
                        sh = 0 if len(v.shape) == 1 else v.shape[1]
                        shape_dict[k].append(sh)
        skip_attrs = [k for k, v in type_dict.items() if not len(set(v)) == 1 or not len(set(shape_dict[k])) == 1]

        # type_dict = defaultdict(list)
        # shape_dict = defaultdict(list)
        # files = []
        # for subset in self.deck.get_leaves():
        #     logging.info(f"Collecting metadata from {subset.longname()}")
        #     files.append(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_RNA_imputed.loom"))
        #     with loompy.connect(os.path.join(self.config.paths.build, subset.longname(), subset.longname() + "_RNA_imputed.loom"), mode="r") as ds:
        #         for k, v in ds.ca.items():
        #             type_dict[k].append(v.dtype)
        #             sh = 0 if len(v.shape) == 1 else v.shape[1]
        #             shape_dict[k].append(sh)
        # skip_attrs = [k for k, v in type_dict.items() if not len(set(v)) == 1 or not len(set(shape_dict[k])) == 1]

        logging.info(f"Collecting all cells into {out_file}")
        loompy.combine_faster(files, out_file, selections=selections, key="Accession", skip_attrs=skip_attrs)
        
        logging.info(f'Transferring attritubes from peaks to RNA')
        with loompy.connect(self.peak_file) as ds:
            with loompy.connect(out_file) as dsr:            
                transfer_ca(ds,dsr, 'CellID')
        
        with loompy.connect(self.peak_file) as ds:
            RA = RNA_analysis(ds, outdir=os.path.join(self.config.paths.build, 'Pool'))
            RA.annotate()
            
        with loompy.connect(self.peak_agg) as dsagg:
            with loompy.connect(os.path.join(self.config.paths.build, 'Pool', 'Pool_RNA.agg.loom')) as dsr:
                dsr.attrs.linkage = dsagg.attrs.linkage

        with loompy.connect(self.peak_file) as ds:
            RA.generate_plots()

    def Aggregate_motifs(self) -> None:
        logging.info(f'Start motif aggragation and plotting')
        with loompy.connect(self.peak_file) as ds:
            MA = motif_aggregator('Pool')
            MA.fit()

    def export_bigwigs(self) -> None:
        logging.info('Exporting Bigwigs')
        bigwig_dir = os.path.join(self.config.paths.build, 'Pool', 'bigwigs')
        if not os.path.isdir(bigwig_dir):
            os.mkdir(bigwig_dir)

        ## Export bigwigs by cluster
        with loompy.connect(self.peak_file, 'r') as ds:
            with mp.get_context().Pool(min(mp.cpu_count(),10)) as pool:
                for cluster in np.unique(ds.ca.Clusters):
                    cells = [x.split(':') for x in ds.ca['CellID'][ds.ca['Clusters'] == cluster]]
                    pool.apply_async(export_bigwig, args=(cells, self.config.paths.samples, bigwig_dir, cluster,))
                pool.close()
                pool.join()
        with loompy.connect(self.peak_agg) as ds:
            if 'ClusterName' in ds.ca:
                for i in range(ds.shape[1]):
                    f = os.path.join(bigwig_dir, f'cluster_{i}.bw')
                    f2 = os.path.join(bigwig_dir, f"{ds.ca.ClusterName[i].replace(' ', '_').strip('.')}.bw")
                    os.rename(f,f2)
        logging.info(f'Finished saving bigwigs')