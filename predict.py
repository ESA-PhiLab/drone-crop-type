"""Predict crop types of drone images using trained CNNs

Before using this script, you should have
* a config file in yaml format (see below)
* two trained keras models (one for cropland identification and one for crop types)
* two free GPUs available (e.g. use export CUDA_VISIBLE_DEVICES=0,1)

How does it work? For each mosaic, the cropland model will predict a score (0: no-cropland, 1: cropland).
If the score is higher than cropland-model:threshold, the croptypes model predicts the crop type for it.

Usage:
    > python predict.py CONFIG_FILE.yaml

Please provide your own CONFIG_FILE in yaml format. It should look like this:

```
mosaics: /path/to/mosaics/m*.tif
output_path: /path/to/empty/folder
patch_size: 299
config['stride']: 299
cropland_model:
    path: /path/to/model/file
    preprocess_input: 'vgg16'
    # Threshold of 0.6, i.e. we rather throw away real
    # cropland samples than keeping non-cropland ones
    threshold: 0.6
    batch_size: 32
croptypes_model:
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
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import model_from_json, load_model
from tensorflow.keras.utils import Sequence
import geopandas
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import Polygon
import yaml

import models

def create_model(path, gpu=0):
    print(f'[GPU {gpu}] Loading model from', path)
    with tf.device(f'/device:GPU:{gpu}'):
        g = tf.Graph()
        with g.as_default():
            with open(join(path, 'model.json'), 'r') as json_file:
                model = model_from_json(json.load(json_file))
            model.load_weights(join(config['cropland_model']['path'], 'checkpoint'))
    return g, model

def add_predictions(df, predictions, mosaic):
    sorted_indices = np.argsort(predictions, axis=1)
    
    for i in range(batch_predictions.shape[0]):
        window = windows[i]
        polygon = Polygon((
            mosaic.xy(window.row_off, window.col_off),
            mosaic.xy(window.row_off, window.col_off + window.width),
            mosaic.xy(window.row_off + window.height, window.col_off + window.width),
            mosaic.xy(window.row_off + window.height, window.col_off),
        ))
        
        label_1 = config['classes'][sorted_indices[i, -1]]
        label_2 = config['classes'][sorted_indices[i, -2]]
        confidence_1 = predictions[i, sorted_indices[i, -1]]
        confidence_2 = predictions[i, sorted_indices[i, -2]]
        
        df = df.append({
            'column': window.col_off,
            'row': window.row_off,
            'label_1': label_1,
            'conf_1': confidence_1,
            'label_2': label_2, 
            'conf_2': confidence_2,
            'geometry': polygon,
        }, ignore_index=True)
        
    return df

config_file = sys.argv[1]
with open(config_file) as file:
    config = yaml.safe_load(file)
    
os.makedirs(config['output_path'], exist_ok=True)
    
# Available gpus:
gpus = tf.config.experimental.list_logical_devices('GPU')
cropland_graph, cropland_model = create_model(config['cropland_model']['path'], gpu=0)
# croptypes_graph, croptypes_model = create_model(config['croptypes_model']['path'], gpu=1)
    
for mosaic_file in glob(config['mosaics']):
    mosaic_id = Path(mosaic_file).stem

    print('Processing', mosaic_id)

    # Load cropland model
    ...

    predictions = geopandas.GeoDataFrame(
        columns=['LC', 'CODE', 'CONF_1', 'LC_2', 'CONF_2', 'CROP', 'CropType', 'CropType_l', 'COLUMN', 'ROW'],
    )
    with rio.open(mosaic_file) as mosaic:
        image = np.moveaxis(mosaic.read(), 0, -1)
        
    print(image.shape)

    n_x = (image.shape[0] - config['patch_size']) // config['stride'] + 1
    n_y = (image.shape[1] - config['patch_size']) // config['stride'] + 1
    step_x = config['stride']
    step_y = config['stride']
    # pre-allocated for efficiency
    patches = np.empty((
                            n_x * n_y,
                            config['patch_size'],
                            config['patch_size'],
                            3
                            ))
    patch_ids = []
    num_valid_patches = 0
    for col in range(n_x):
        for row in range(n_y):
            region = slice(col * step_x, col * step_x + config['patch_size']), \
                     slice(row * step_y, row * step_y + config['patch_size']), ...
            patch = image[region]
            if 0 in patch.shape or 0 in patch[..., 3]:
                continue

            # Get rid of alpha!
            patches[num_valid_patches, ...] = patch[..., :-1]
            patch_ids.append((col, row))
            num_valid_patches += 1

    patches = patches[:num_valid_patches, ...]
    print(patches.shape)
    with tf.device(f'/device:GPU:0'):
        with cropland_graph.as_default():
            cropland_predictions = cropland_model.predict(patches)# > config['cropland_model']['threshold']

    crops = cropland_predictions < config['cropland_model']['threshold']
    print(f'{mosaic_id} {int(crops.mean()*100)}% are crops')
    np.save(join(config['output_path'], mosaic_id+"_cropland.npy"), cropland_predictions)
#     np.save()
#     sys.exit()
    # predictions for croptypes:
#     with tf.device(f'/device:GPU:0'):
#         with croptypes_graph.as_default():
#             croptyes_predictions = croptypes_model.predict(
#                 patches[cropland_predictions]
#             )
#     predictions = add_predictions(
#         predictions, batch_predictions, mosaic, windows
#     )
#     predictions_filename = 
#     os.makedirs(dirname(predictions_filename), exist_ok=True)
#     predictions.to_file(predictions_filename)