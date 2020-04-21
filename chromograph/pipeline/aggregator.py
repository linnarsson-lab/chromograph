  
import logging
from typing import Dict, List, Union

import numpy as np
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist

import loompy
from cytograph.annotation import AutoAnnotator, AutoAutoAnnotator
from cytograph.enrichment import FeatureSelectionByMultilevelEnrichment, Trinarizer
from cytograph.manifold import GraphSkeletonizer

from chromograph.pipeline import config

class Aggregator:
	def __init__(self, *, f: Union[float, List[float]] = 0.2, mask: np.ndarray = None) -> None:
		self.f = f
		self.mask = mask
        self.config = config.load_config()

	def aggregate(self, ds: loompy.LoomConnection, *, out_file: str, agg_spec: Dict[str, str] = None) -> None:
        cells = ds.col_attrs["Clusters"] >= 0
		labels = ds.col_attrs["Clusters"][cells]
		n_labels = len(set(labels))