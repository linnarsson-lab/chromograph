## Split Test

import os, sys
import logging
import loompy
import numpy as np
from scipy.cluster.hierarchy import cut_tree
import matplotlib.pyplot as plt
from cytograph.plotting.colors import colorize
from cytograph.pipeline import Tempname
import networkx as nx
import community
import shutil

import chromograph
from chromograph.pipeline import config

def calc_cpu(n_cells):
    n = np.array([1e2, 1e3, 1e4, 1e5, 5e5, 1e6, 2e6])
    cpus = [1, 3, 7, 14, 28, 28, 56]
    idx = (np.abs(n - n_cells)).argmin()
    return cpus[idx]


def split_subset(config, subset: str, method: str = 'coverage', thresh: float = None, python_exe=None, min_RNA:int=None) -> None:

    loom_file = os.path.join(config.paths.build, subset, subset + "_peaks.loom")
    out_dir = os.path.join(config.paths.build, subset, "exported", method)

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    with Tempname(out_dir) as exportdir:
        with loompy.connect(loom_file) as ds:

            if method == 'dendrogram':

                logging.info("Splitting by dendrogram")

                # split dendrogram into two and get new clusters
                agg_file = os.path.join(config.paths.build, subset, subset + "_peaks.agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]
                clusters = branch[ds.ca.Clusters]

                # save split attribute and plot
                ds.ca.Split = clusters
                names, labels = np.unique(ds.ca.Split, return_inverse=True)

                os.mkdir(exportdir)
                plt.figure(None, (16, 16))
                colors = colorize(names)
                plt.scatter(ds.ca[config.params.main_emb][:, 0], ds.ca[config.params.main_emb][:, 1], c=colors[labels], s=5)
                plt.axis('off')
                def h(c):
                    return plt.Line2D([], [], color=c, ls="", marker="o")
                plt.legend(handles=[h(colors[i]) for i in range(len(names))], labels=list(names), loc='lower left', markerscale=1, frameon=False, fontsize=10)
                plt.title(f"Force split", fontsize=20)

                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'coverage':

                # set thresh to 0.98 if not specified
                if thresh is None:
                    thresh = 0.98

                logging.info(f"Splitting by dendrogram if coverage is above {thresh}")

                # Get dendrogram from agg file and split into two
                logging.info("Splitting dendrogram in .agg file")
                agg_file = os.path.join(config.paths.build, subset, subset + "_peaks.agg.loom")
                with loompy.connect(agg_file, 'r') as dsagg:
                    Z = dsagg.attrs.linkage
                    branch = cut_tree(Z, 2).T[0]

                # Assign clusters based on the dendrogram cut
                clusters = branch[ds.ca.Clusters]

                # Check cluster sizes
                total = len(clusters)
                if np.any(np.bincount(clusters) / total < 0.01):
                    logging.info(f"A cluster is too small.")
                    return False, []

                # Load KNN graph
                logging.info("Loading KNN graph")
                G = nx.from_scipy_sparse_matrix(ds.col_graphs.KNN)

                # Calculate coverage of this partition on the graph
                logging.info("Calculating coverage of this partition")
                partition = []
                for c in np.unique(clusters):
                    partition.append(set(np.where(clusters == c)[0]))
                cov = nx.algorithms.community.quality.coverage(G, partition)

                # Stop if coverage is below thresh
                ds.attrs.Coverage = cov
                logging.info(f"Coverage threshold set at {thresh}")
                if cov < thresh:
                    logging.info(f"Partition is not separable: {cov:.5f}.")
                    return False, []

                ## Get counts of partitions and stop if one partition is too small
                p1 = np.sum(clusters==0)
                p2 = np.sum(clusters==1)
                min_cells = 10 * config.params.min_cells_cluster
                if (p1 < min_cells) or (p2 < min_cells):
                    logging.info(f"Not enough cells in partitions ({p1}, {p2})")
                    return False, []

                if min_RNA:
                    r1 = len(np.where((ds.ca.Chemistry=='multiome_atac')&(clusters==0))[0]) / p1
                    r2 = len(np.where((ds.ca.Chemistry=='multiome_atac')&(clusters==1))[0]) / p2
                    if (r1 < min_RNA) or (r2 < min_RNA):
                        logging.info(f"Fraction multiome cells in partitions too low ({r1}, {r2})")
                        return False, []                    

                # Otherwise save split attribute and plot
                ds.ca.Split = clusters
                logging.info(f"Partition is separable: {cov:.5f}.")
                logging.info(f"Plotting partition")
                names, labels = np.unique(ds.ca.Split, return_inverse=True)
                sizes = np.bincount(ds.ca.Split)

                plt.figure(None, (16, 16))
                colors = colorize(names)
                plt.scatter(ds.ca[config.params.main_emb][:, 0], ds.ca[config.params.main_emb][:, 1], c=colors[labels], s=5)
                plt.axis('off')
                def h(c):
                    return plt.Line2D([], [], color=c, ls="", marker="o")
                plt.legend(handles=[h(colors[i]) for i in range(len(names))], labels=list([str(name) + ':' + str(n) for name, n in zip(names, sizes)]), loc='lower left', markerscale=1, frameon=False, fontsize=10)
                plt.title(f"Coverage: {cov:.5f}", fontsize=20)

                os.mkdir(exportdir)
                plt.savefig(os.path.join(exportdir, "Split.png"), dpi=150)
                plt.close()

            if method == 'cluster':

                logging.info("Splitting by clusters.")
                clusters = ds.ca.Clusters
                ds.ca.Split = clusters

        # Calculate split sizes
        sizes = np.bincount(clusters)
        logging.info("Creating punchcard")
        with open(f'{config.paths.build}/punchcards/{subset}.yaml', 'w') as f:
            for i in np.unique(clusters):
                # Calc cpu and memory
                n_cpus = calc_cpu(sizes[i])
                memory = 750 if n_cpus == 56 else config.execution.memory
                # Write to punchcard
                name = chr(i + 65) if i < 26 else chr(i + 39) * 2
                f.write(f'{name}:\n')
                f.write('  include: []\n')
                f.write(f'  onlyif: Split == {i}\n')
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
                name = chr(i + 65) if i < 26 else chr(i + 39) * 2
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

        return True, file