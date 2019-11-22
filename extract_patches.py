from glob import glob
from random import random
import os
from os.path import basename

from ai4eo.preprocessing import extract_patches
import geopandas
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from skimage.io import imsave

