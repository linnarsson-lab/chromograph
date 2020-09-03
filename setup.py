from setuptools import setup, find_packages

__version__ = "0.0.0"
exec(open('chromograph/_version.py').read())

setup(
	name="chromograph",
	version=__version__,
	packages=find_packages(),
	install_requires=[
		'loompy',
		'numpy',
		'scikit-learn',
		'scipy==1.4.1',
		'matplotlib',
		'networkx',
		'python-louvain',  # is imported as "community"
		'hdbscan',
		'pyyaml',
		'statsmodels',  # for multiple testing
		'numpy-groupies==0.9.6',
		#'numba=0.49.1', ## Version is important, others throw errors
		'tqdm',
		'umap-learn==0.4.6',  # imported as "umap"
		'torch',
		'harmony-pytorch',
		'pynndescent==0.4.8',
		'click',
		'leidenalg',
		'unidip',
		'opentsne',
        'pybedtools',
        'macs2',
		'pygenometracks',
		'fisher',
		'kneed'
		## Install through bioconda ucsc-bedgraphtobigwig ucsc-genepredtobed ucsc-gtftogenepred ucsc-bigwigaverageoverbed
		],
	include_package_data=True,
	author="Linnarsson Lab",
	author_email="camiel.mannens@ki.se",
	description="Pipeline for single-cell ATAC-seq analysis",
	license="MIT",
	url="https://github.com/linnarsson-lab/chromograph",
)