# chromograph

## Currently contains
1. Chromgen: Binning and Gene-Activity Scoring of Cell ranger output
2. Bin analysis: TF-IDF normalization, Incremental PCA, Harmony batch correction, KNN/mKNN/RNN network, louvain clustering and tSNE/UMAP
3. Peak calling per cluster: parallelized MACS calling and HOMER annotation/motif enrichment of peaks. BigWig output for easy visualization
4. Peak Analysis: HPF factorization, KNN/mKNN/RNN network, louvain clustering and tSNE/UMAP
5. Gene Activity: Aggregation, marker identification and graph skeletonization
6. Motifs: Motif scoring based on HOCOMOCO and aggregation/normalization by cluster


## Installation instruction

We have only used this on a linux system. Most likely this will not work correctly on windows.
Installation should take around 10 minutes.

Create a new environments:
```
conda create --name chromo python=3.7
```

Make sure to install the right versions of pynndescent, numba and umap since we will install cytograph before chromograph
```
pip install pynndescent==0.4.8 umap-learn==0.4.6 numba==0.49.1
```

Next git clone cytograph-dev and move into the folder
```
git clone https://github.com/linnarsson-lab/cytograph-dev.git
cd cytograph-dev
pip install -e .
cd ..
```

Next we install some bioconda packages from ucsc
```
conda install ucsc-bedgraphtobigwig ucsc-genepredtobed ucsc-gtftogenepred ucsc-bigwigaverageoverbed cudatoolkit libllvm8 libllvm9 llvm-openmp
```

Then we can clone chromograph and install it 
```
git clone https://github.com/linnarsson-lab/chromograph.git
cd chromograph
pip install -e .
```

Next it is advisable to set the default paths for where to find the samples and external tools. Create a file file called .chromograph in the root folder. Replace my paths with your own:
```
paths:
  samples: "/datb/sl/camiel/scATAC/samples"
  cell_ranger: "/data/proj/chromium"
  qc: "/datb/sl/camiel/scATAC/qc_plots"
  RNA: "/proj/loom"
  bedtools: "/data/bin/bedtools2/bin/"
  ref: "/data/ref/cellranger-atac/refdata-cellranger-atac-GRCh38-1.2.0/"
  metadata: "/data/www-sanger/10X/DB/sqlite3_chromium.db"
  MACS: "/home/camiel/anaconda3/envs/chromo/bin/macs2"
  HOMER: "/data/bin/homer/bin"
  autoannotation: "/home/camiel/auto-annotation/Human_dev"
```

## Preprocess data

Use the Chromgen function from chromograph.preprocessing to generate binned matrices and separate fragment pile-ups by cell

## Creating a build

Create a new folder in which to start your build. It should contain the following files:

config.yaml. Any changes in this file override the default paths you defined earlier:
```
paths:
  metadata: "/nfs/sanger-data/10X/DB/sqlite3_chromium.db"
  samples: '/datb/sl/camiel/scATAC/samples'

params:
  skip_missing_samples: True
  Normalization: TF-IDF
  factorization: ['SVD']
  doublets_action: remove
  batch_size : 512
  batch_keys: ["Chemistry", "Shortname"]
  bin_size: 20000
  UMAP: True
  peak_size: 400
  poisson_pooling: True
  skip_attrs: ["Cellconc", "Targetnumcells"] ## Attributes will not be copied from individual loom files

execution:
  n_cpus: 52

steps: [bin_analysis, peak_calling, peak_analysis, prom, RNA, Impute_RNA, motifs, bigwigs, split] ## Remove steps you don't want to run
```

create a folder 'punchcards'. With the following file that contains the sample IDs.
punchcards/Root.yaml:
```
All:
 include: ['your samples']
```
You can now run the main_workflow from chromograph.pipeline to create a new build. This will generate a new folder 'All' containing Loom files, a peaks folder (with the consensus peak set and individuals peaksets from the broad subsets) and the 'exported' folder with plots. Depending on the size of the dataset this can take a long time (many hours). Additionally some steps here can take a lot of working memory, so it might not work on a desktop computer if you want to analyze large datasets (e.g. > 100k cells).

```
python ~/chromograph/chromograph/pipeline/main_workflow.py
```

The next step is to run subset_workflow.py. This is done in a similar way as the main_workflow, except that the name of the subset must be defined and will generate all the downstream analysis.
Depending on the size of the dataset this can take a long time (many hours).
For instance, to finish up the analysis of the full collection of data the following command can be used:

```
python ~/chromograph/chromograph/pipeline/subset_workflow.py All
```

If you want to run smaller subsets of the data you must add a new file to the punchcards folder. For instance to run subsets of the 'All' dataset described in the fetal multiomics paper the following punchcard can be added, assuming a column attribute 'Class' has been added to annotate the subsets.
punchcards/All.yaml:
```
Immune:
  include: []
  onlyif: Class == Immune
  execution:
    n_cpus: 28
    memory: 256
FBL:
  include: []
  onlyif: Class == Fibroblast
  execution:
    n_cpus: 28
    memory: 256
Vascular:
  include: []
  onlyif: Class == Vascular
  execution:
    n_cpus: 28
    memory: 256
RGL:
  include: []
  onlyif: Class == Radial_glia
  execution:
    n_cpus: 52
    memory: 256
OPC:
  include: []
  onlyif: Class == Oligo
  execution:
    n_cpus: 28
    memory: 256
Neuron:
  include: []
  onlyif: Class == Neuron
  execution:
    n_cpus: 52
    memory: 256
```

If you would like to further split any of the leafs in this subset analysis that can be done by creating a punchcard named All_yoursubset.yaml. We subdivided the Neuron leaf further using the following punchcard. After annotating the data.
All_Neuron.yaml
```
GABA:
  onlyif: Clusters in 7,8,9,14,15,16,17,18,19,23,24,25,26,27,28

GLUT:
  onlyif: Clusters in 0,1,2,3,4,5,6,10,11,12,13,20,22

Peptidergic:
  include: []
  onlyif: Clusters == 21
```

When you are done annotating the subsets in more details and would like to pool the data again to analyze it with better annotation and more course clustering, you can use the Pool_workflow. This will aggregate all the end leafs and generate a new folder called pool. If any clusters are manually labelled as 'unclear' in the ClusterName (capitals matter here) they will be excluded.

```
python ~/chromograph/chromograph/pipeline/Pool_workflow.py
```
