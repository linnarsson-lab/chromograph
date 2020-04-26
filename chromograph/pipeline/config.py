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
			"cell_ranger": "",
			"bedtools": "",
			"MACS": "",
			"HOMER": "",
			"ref": "",
			"autoannotation": "",
			"metadata": "",
			"fastqs": "",
			"index": "",
			"qc": ""
		}),
		"params": Config(**{
			"batch_keys": [],  # Set to empty list for no batch correction, or use e.g. ["Chemistry"]
			"k": 25,
			"batch_size": 512,
			"k_pooling": 10,
			"Normalization": "TF-IDF", 
			"factorization": "PCA",  # or "HPF" or "both"
			"n_factors": 40,
			"HPF_factors": 48,
			"level": 5000,
			"cov": 1.75,
			"bin_size": 5000,
			"bin_quantile": 0.75,
			"peak_quantile": 0.75,
			# "doublets_action": "remove",
			# "mask": ("cellcycle", "sex", "ieg", "mt"),
			"min_fraction_good_cells": 0.4,
			"max_fraction_MT_genes": 0.05,
			# "max_doubletFinder_TH": 0.4,
			"skip_missing_samples": False,
			"skip_metadata": False,
			"passedQC": False,
			"clusterer": "louvain",  # or "surprise"
			"nn_space": "auto",
			"peak_depth": 2.5e7,
			"peak_min_cells": 50,
			"reference_assembly": "GRCh38"
		}),
		# "steps": ("doublets", "poisson_pooling", "nn", "embeddings", "clustering"),
		"steps": ("nn", "embeddings", "clustering"),
		"execution": Config(**{
			# "n_cpus": available_cpu_count(),
			"n_cpus": 52,
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