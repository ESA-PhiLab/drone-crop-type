"""Predict crop types of drone images using trained CNNs

Before using this script, you should have
* a config file in yaml format (see below)
* two trained keras models (one for cropland identification and one for crop types)
* two free GPUs available (e.g. use export CUDA_VISIBLE_DEVICES=0,1)

How does it work? For each mosaic, the cropland model will predict a score (0: no-cropland, 1: cropland).
If the score is higher than cropland-model:threshold, the croptypes model predicts the crop type for it.

Usage:
    > python predict.py mode config_file

Args:
    mode:
        Can be cropland or croptypes. Cropland should be run first.
    config_file:
        Please provide your own PREDICT_CONFIG file in yaml format. It should look like the example in predict.yaml.

Example:
    To classify all mosaics, run this:
    > export CUDA_VISIBLE_DEVICES=0
    > python predict.py cropland predict.yaml
    > python predict.py croptypes predict.yaml
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
    #with tf.device(f'/device:GPU:{gpu}'):
    g = tf.Graph()
    with g.as_default():
        with open(join(path, 'model.json'), 'r') as json_file:
            model = model_from_json(json.load(json_file))
        model.load_weights(join(path, 'checkpoint'))
    return g, model

def to_geopandas(cols_rows, predictions, mosaic):
    sorted_indices = np.argsort(predictions, axis=1)
    
    df = [None] * predictions.shape[0]
    for i in range(predictions.shape[0]):
        col, row = cols_rows[i]
        polygon = Polygon((
            mosaic.xy(row*config['patch_size'], col*config['patch_size']),
            mosaic.xy(row*config['patch_size'], (col+1)*config['patch_size']),
            mosaic.xy((row+1)*config['patch_size'], (col+1)*config['patch_size']),
            mosaic.xy((row+1)*config['patch_size'], col*config['patch_size']),
        ))
        
        label_1 = config['croptypes_model']['classes'][sorted_indices[i, -1]]
        label_2 = config['croptypes_model']['classes'][sorted_indices[i, -2]]
        confidence_1 = predictions[i, sorted_indices[i, -1]]
        confidence_2 = predictions[i, sorted_indices[i, -2]]
        
        df[i] = {
            'COLUMN': cols_rows[i][0],
            'ROW': cols_rows[i][1],
            'LC': label_1,
            'CONF_1': confidence_1,
            'LC_2': label_2, 
            'CONF_2': confidence_2,
            'geometry': polygon,
            'CropType': 1,
            'CropType_l': 'Cultivated Field'
        }
        
    return geopandas.GeoDataFrame(df)

mode = sys.argv[1]
config_file = sys.argv[2]
with open(config_file) as file:
    config = yaml.safe_load(file)
    
os.makedirs(config['output_path'], exist_ok=True)
    
if mode == "cropland":
    cropland_graph, cropland_model = create_model(config['cropland_model']['path'])
elif mode == "croptypes":
    croptypes_graph, croptypes_model = create_model(config['croptypes_model']['path'])
else:
    raise ValueError('Mode must be either cropland or croptypes!')
    
for mosaic_file in glob(config['mosaics']):
    mosaic_id = Path(mosaic_file).stem

    print('Processing', mosaic_id)

#     predictions = geopandas.GeoDataFrame(
#         columns=['LC', 'CONF_1', 'LC_2', 'CONF_2', 'CROP', 'CropType', 'CropType_l', 'COLUMN', 'ROW'],
#     )
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

        patch_ids = np.array(patch_ids)
        patches = patches[:num_valid_patches, ...]
        if mode == "cropland":
            with cropland_graph.as_default():
                cropland_predictions = np.squeeze(cropland_model.predict(patches))

            cropland = cropland_predictions < config['cropland_model']['threshold']
            np.save(join(config['output_path'], mosaic_id+"_cropland.npy"), cropland_predictions)
        else:
            cropland_predictions = np.squeeze(np.load(join(config['output_path'], mosaic_id+"_cropland.npy")))
            cropland = cropland_predictions < config['cropland_model']['threshold']
        print(f'{mosaic_id} {int(cropland.mean()*100)}% are crops')

        # predictions for croptypes:
        if mode == "croptypes":
            print(cropland.shape)
            with croptypes_graph.as_default():
                croptyes_predictions = croptypes_model.predict(
                    patches[cropland, ...]
                )
            crop_types = np.argmax(croptyes_predictions, axis=-1)
            np.save(join(config['output_path'], mosaic_id+"_croptypes.npy"), crop_types)
            print(config['croptypes_model']['classes'])
            print(np.unique(crop_types, return_counts=True))
            predictions = to_geopandas(
               patch_ids[cropland], croptyes_predictions, mosaic
            )
            predictions_filename = join(config['output_path'], mosaic_id+"_croptypes.shp")
            predictions.to_file(predictions_filename)