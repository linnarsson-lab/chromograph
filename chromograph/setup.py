from setuptools import setup, find_packages

__version__ = "0.0.0"
exec(open('_version.py').read())

setup(
	name="chromograph",
	version=__version__,
	packages=find_packages(),
	install_requires=[
		'loompy',
		'numpy',
		'scikit-learn',
		'scipy',
		'networkx',
		'python-louvain',  # is imported as "community"
		'hdbscan',
		'pyyaml',
		'statsmodels',  # for multiple testing
		'numpy-groupies==0.9.6',
		'tqdm',
		'umap-learn',  # imported as "umap"
		'torch',
		'harmony-pytorch',
		'pynndescent',
		'click',
		'leidenalg',
		'unidip',
		'opentsne',
        'pybedtools',
        'cytograph',
        'macs2',
	],
	include_package_data=True,
	author="Linnarsson Lab",
	author_email="camiel.mannens@ki.se",
	description="Pipeline for single-cell ATAC-seq analysis",
	license="MIT",
	url="https://github.com/linnarsson-lab/chromograph",
)