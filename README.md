# chromograph

## Currently contains
1. Chromgen: Binning and Gene-Activity Scoring of Cell ranger output
2. Bin analysis: TF-IDF normalization, Incremental PCA, Harmony batch correction, KNN/mKNN/RNN network, louvain clustering and tSNE/UMAP
3. Peak calling per cluster: parallelized MACS calling and HOMER annotation/motif enrichment of peaks. BigWig output for easy visualization
4. Peak Analysis: HPF factorization, KNN/mKNN/RNN network, louvain clustering and tSNE/UMAP
5. Gene Activity: Aggregation, marker identification and graph skeletonization
6. Motifs: Motif scoring based on HOCOMOCO and aggregation/normalization by cluster

## To do:
* Peak / gene enrichment statistics
* Lineage inference
* Feature selection methods
* Motif enrichments
* GWAS integration
* Enhancer prediction


## Installation instruction

Create a new environments:
`
conda create --name chromo python=3.7
`

Make sure to install the right versions of pynndescent, numba and umap since we will install cytograph before chromograph
`
pip install nndescent==0.3.3 umap-learn==0.3.9 numba==0.49.1
`

Next git clone cytograph-dev and move into the folder
`
git clone https://github.com/linnarsson-lab/cytograph-dev.git
cd cytograph-dev
pip install -e .
cd ..
`

Next we install some bioconda packages from ucsc
`
conda install ucsc-bedgraphtobigwig ucsc-genepredtobed ucsc-gtftogenepred ucsc-bigwigaverageoverbed
`

Then we can clone chromograph and install it 
`
git clone https://github.com/linnarsson-lab/chromograph.git
cd chromograph
pip install -e .
`
