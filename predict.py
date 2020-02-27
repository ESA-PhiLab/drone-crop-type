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
from pathlib import Path

from tensorflow.keras.models import from_json, load_model
from tensorflow.keras.utils import Sequence
import geopandas
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import Polygon
import yaml

import models

class BigImageLoader(Sequence):
    def __init__(self, image):
        self.image = rio.open(mosaic_file)
        
    def __len__(self):
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        if self._weights is None:
            sample_ids = \
                self._indices[idx*self.batch_size:(idx+1)*self.batch_size]
        return self.get_samples(sample_ids)
        
    def get_batch(self, ):
        nx_windows = (self.image.width // input_size) + 1
        ny_windows = (self.image.height // input_size) + 1
        batch = []
        windows = []
        for x in range(x_windows):
            for y in range(y_windows):
                window = Window(input_size*x, input_size*y, input_size, input_size)
                image = mosaic.read(window=window)
                # Missing values are indicated as a 0 value in the alpha channel
                if 0 in image.shape or 0 in image[3, :]:
                    continue
                # We need to preprocess the image first:
                image = self._preprocess_input(image[:3, ...].T)
                batch.append(image)
                windows.append(window)
                if len(batch) >= batch_size:
                    yield np.array(batch)
                    batch.clear()
                    windows.clear()

# join(model_dir, 'checkpoint')

config_file = sys.argv[1]
with open(config_file) as file:
    config = yaml.safe_load(file)
    
for mosaic_file in glob(config['mosaics']):
    mosaic_id = Path(mosaic_file).stem
    
    print('Processing', mosaic_id)
    
    with rio.open(mosaic_file) as mosaic:
        predictions = geopandas.GeoDataFrame(
            columns=['LC', 'CODE', 'CONF_1', 'LC_2', 'CONF_2', 'CROP', 'CropType', 'CropType_l', 'COLUMN', 'ROW'],
        )
        x_windows = (mosaic.width // input_size) + 1
        y_windows = (mosaic.height // input_size) + 1
        batch = []
        windows = []
        for x in range(x_windows):
            for y in range(y_windows):
                window = Window(input_size*x, input_size*y, input_size, input_size)
                image = mosaic.read(window=window)
                # Missing values are indicated as a 0 value in the alpha channel
                if 0 in image.shape or 0 in image[3, :]:
                    continue
                # We need to preprocess the image first:
                image = preprocess_input(image[:3, ...].T)
                batch.append(image)
                windows.append(window)
                if len(batch) >= batch_size:
                    batch_predictions = model.predict_on_batch(np.array(batch))
                    predictions = add_predictions(
                        predictions, batch_predictions, mosaic, windows
                    )
                    batch.clear()
                    windows.clear()
    except:
        print('error with', mosaic_id)

    predictions_filename = config['predictions_path'].format(model=model_name, mosaic=mosaic_id)
    os.makedirs(dirname(predictions_filename), exist_ok=True)
    predictions.to_file(predictions_filename)
