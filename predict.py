"""Predict crop types of drone images using trained CNNs

Before using this script, you should have
* a config file in yaml format (see below)
* two trained keras models (one for cropland identification and one for crop types)

How does it work? For each mosaic, the cropland model will predict a score (0: no-cropland, 1: cropland). 
If the score is higher than cropland-model:threshold, the croptypes model predicts the crop type for it.

Usage:
    > python predict.py CONFIG_FILE.yaml
    
Please provide your own CONFIG_FILE in yaml format. It should look like this:

```
mosaics: /path/to/mosaics/m*.tif
output_path: /path/to/empty/folder
input_shape: [224, 224]
cropland-model:
    path: /path/to/model/file
    preprocess_input: 'vgg16'
    # Threshold of 0.6, i.e. we rather throw away real 
    # cropland samples than keeping non-cropland ones 
    threshold: 0.6
    batch_size: 32
croptypes-model:
    path: /path/to/model/file
    preprocess_input: 'vgg16'
    batch_size: 32
    classes: ['cassava', 'groundnut', 'maize', 'tobacco']
```
"""

from glob import glob
import json
import os
from os.path import join, dirname, basename
import sys

from tensorflow.keras.models import from_json, load_model
import geopandas
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import Polygon
import yaml

import models

# join(model_dir, 'checkpoint')

config_file = sys.argv[1]
with open(config_file) as file:
    config = yaml.safe_load(file)
    
for mosaic_file in glob(config['mosaics']):
    with rio.open(mosaic_file) as mosaic:
        
    
