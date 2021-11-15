## Imports
import loompy
import matplotlib.pyplot as plt
import logging
import numpy as np
from chromograph.cicero.cicero import *
from chromograph.cicero.generate_Coaccessibilty_networks import generate_Coaccessibilty_networks
import sys

f = sys.argv[1]
generate_GA = bool(sys.argv[2])

with loompy.connect(f) as ds:
    generate_Coaccessibilty_networks(ds, generate_Gene_Activity=generate_GA)