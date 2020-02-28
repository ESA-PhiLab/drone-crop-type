"""Create fold files for training and test images
"""

import numpy as np
import pandas as pd
from typhon.files import FileSet

df = pd.concat([
    FileSet('/scratch-a/jmrziglod/sen2agri/data/malawi_summer/patches/original/{label}/*.png').to_dataframe(),
    FileSet('/scratch-a/jmrziglod/sen2agri/data/malawi_summer/patches/augmented/{label}/*.png').to_dataframe()
])

unique_ids = np.unique(df.id.values)
shuffled_ids = np.random.choice(unique_ids, size=len(unique_ids), replace=False)
ratio = 0.1
test_ids = shuffled_ids[:int(shuffled_ids.size*ratio)]
train_ids = shuffled_ids[int(shuffled_ids.size*ratio):]

with open('/home/jmrziglod/projects/sen2agri/drone-crop-type/folds/malawi_summer/train_test_all_mosaics/train.txt', 'w') as txt_file:
    txt_file.write("\n".join(df.index[df.id.isin(train_ids)].tolist()))
with open('/home/jmrziglod/projects/sen2agri/drone-crop-type/folds/malawi_summer/train_test_all_mosaics/test.txt', 'w') as txt_file:
    txt_file.write("\n".join(df.index[df.id.isin(test_ids)].tolist()))