## Imports
import loompy
import matplotlib.pyplot as plt
import logging
import numpy as np
from chromograph.cicero.cicero import *
from chromograph.cicero.generate_Coaccessibilty_networks import generate_Coaccessibilty_networks
import sys

import logging

logger = logging.getLogger()
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

f = sys.argv[1]
fagg = f.split('.')[0] + '.agg.loom'
generate_GA = bool(sys.argv[2])

with loompy.connect(f) as ds:
   with loompy.connect(fagg) as dsagg:
        generate_Coaccessibilty_networks(ds, dsagg, generate_GA=generate_GA).fit()