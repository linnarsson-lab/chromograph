import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import yaml

# from cytograph.utils import available_cpu_count

# from .punchcards import PunchcardSubset, PunchcardView


def merge_namespaces(a: SimpleNamespace, b: SimpleNamespace) -> None:
	for k, v in vars(b).items():
		if isinstance(v, SimpleNamespace):
			merge_namespaces(a.__dict__[k], v)
		else:
			a.__dict__[k] = v


class Config(SimpleNamespace):
	def to_string(self, offset: int = 0) -> str:
		s = ""
		for k, v in vars(self).items():
			s += "".join([" "] * offset)
			if isinstance(v, SimpleNamespace):
				s += f"{k}:\n{v.to_string(offset + 2)}"
			else:
				s += f"{k}: {v}\n"
		return s

	def merge_with(self, path: str) -> None:
		if not os.path.exists(path):
			raise IOError(f"Config path {path} not found.")

		with open(path) as f:
			defs = yaml.load(f, Loader=yaml.Loader)

		if "paths" in defs:
			merge_namespaces(self.paths, SimpleNamespace(**defs["paths"]))
		if "params" in defs:
			merge_namespaces(self.params, SimpleNamespace(**defs["params"]))
		if "steps" in defs:
			self.steps = defs["steps"]
		if "execution" in defs:
			merge_namespaces(self.execution, SimpleNamespace(**defs["execution"]))


# def load_config(subset_obj: Union[Optional[PunchcardSubset], Optional[PunchcardView]] = None) -> Config:
def load_config() -> Config:
	config = Config(**{
		"paths": Config(**{
			"build": "",
			"samples": "",
			"RNA": "",
			"cell_ranger": "",
			"bedtools": "",
			"MACS": "",
			"HOMER": "",
			"ref": "",
			"autoannotation": "",
			"metadata": "",
			"fastqs": "",
			"index": "",
			"qc": "",
			"cicero_path": "",
			"R": "",
			"pythonexe": "/home/camiel/anaconda3/envs/chromo/bin/python"
		}),
		"params": Config(**{
			"batch_keys": [],  # Set to empty list for no batch correction, or use e.g. ["Chemistry"]
			"skip_attrs": [],
			"plot_attrs": ['SEX', 'Shortname', 'Chemistry', 'Tissue'],
			"k": 25,
			"batch_size": 512,
			"poisson_pooling": True,
			"k_pooling": 10,
			"resolution": 1,
			"Normalization": "TF-IDF", 
			"factorization": "SVD",  # or "PCA"
			"peak_factorization": "LSI", # or HPF
			"feature_selection": "Pearson_Residuals",
			"n_factors": 40,
			"HPF_factors": 48,
			"level": 5000,
			"max_fragments": 100000,
			"bin_size": 5000,
			"peak_size": False,
			"bin_quantile": 0.8,
			"N_peaks_decomp": 20000,
			"peak_fraction": 0.01,
			"f_metric": 'euclidean',
			"UMAP": True,
			"main_emb": 'UMAP',
			"min_umis": 1000,
			"max_fraction_MT_genes": 1,
			"min_fraction_unspliced_reads": 0.1,
			# "doublets_action": "remove",
			# "mask": ("cellcycle", "sex", "ieg", "mt"),
			"max_doubletFinder_TH": 0.4,
			"min_fraction_good_cells": 0.4,
			"min_cells_precluster": 400,
			"min_cells_cluster": 50,
			"N_most_enriched": 6,
			"peak_depth": 2.5e7,
			"peak_min_cells": 150,
			"FR_TSS": 0.2,
			"reference_assembly": "GRCh38",
			"Always_iterative": False,
			"min_split": 15
		}),
		"steps": ("bin_analysis", "peak_calling", "peak_analysis", "Karyotype","RNA", "Impute_RNA", "prom", "motifs", "bigwigs", "cicero", 'split'),
		"execution": Config(**{
			# "n_cpus": available_cpu_count(),
			"n_cpus": 26,
			"n_gpus": 0,
			"memory": 256
		})
	})
	# Home directory
	f = os.path.join(os.path.abspath(str(Path.home())), ".chromograph")
	if os.path.exists(f):
		config.merge_with(f)
	# Set build folder
	if config.paths.build == "" or config.paths.build is None:
		config.paths.build = os.path.abspath(os.path.curdir)
	# Build folder
	f = os.path.join(config.paths.build, "config.yaml")
	if os.path.exists(f):
		config.merge_with(f)
	# Current subset or view
	# if subset_obj is not None:
	# 	if subset_obj.params is not None:
	# 		merge_namespaces(config.params, SimpleNamespace(**subset_obj.params))
	# 	if subset_obj.steps != [] and subset_obj.steps is not None:
	# 		config.steps = subset_obj.steps
	# 	if subset_obj.execution is not None:
	# 		merge_namespaces(config.execution, SimpleNamespace(**subset_obj.execution))

	return config