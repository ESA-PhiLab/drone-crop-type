"""Train a CNN on the patches extracted from the drone mosaics

Usage:
    > python inference.py CONFIG_FILE
"""

import json
import os
from os.path import join
import sys
from time import time
import yaml

# Load the config file:
if len(sys.argv) < 2:
    print('You have to provide a path to the configurations file!')
    exit()

try:
    with open(sys.argv[1]) as config_file:
        config = yaml.safe_load(config_file)
except Exception as e:
    print('Could not load configurations file')
    raise e

# Set the GPU device:
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_device'])

from keras.models import load_model
from keras.applications.xception import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.windows import Window
from skimage.io import imsave, imshow, imread
from typhon.files import FileSet
