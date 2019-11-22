from ai4eo.preprocessing import ImageLoader
from imgaug import augmenters as iaa
import numpy as np
import pandas as pd
from typhon.files import FileSet


__all__ = ['Datasets', 'preprocess_input']


def preprocess_input(x):
    x = x.astype(float)
    x /= 127.5
    x -= 1.
    return x


class Datasets:
    def __init__(
            self, path, classes,
            batch_size=None, preprocess_func=None, random_state=1234, validation_split=0.1,
            testing_split=0.15, balance_batch=True,
    ):

        self.preprocess_input = preprocess_func or preprocess_input
        self.classes = classes
        self.class_weights = None
        self.class_indices = None

        self.patches = FileSet(path).to_dataframe()
        
        # Only the classes which are interesting for us:
        self.patches = self.patches[self.patches.label.isin(classes)]
        
        # We need three datasets: training, validation and testing. We want the
        # testing dataset to be completely independent from the training
        # dataset. Hence, we choose all samples from some mosaics
        # (*testing_mosaics*) for testing purposes only. The validation dataset
        # is simply a partition of the training data but split by polygons, i.e. 
        # training and validation data come never from the same polygons.
        #all_mosaics = set(self.patches['mosaic'].unique())
        #training_mosaics = self.patches['mosaic'].isin(all_mosaics - set(testing_mosaics))
        #testing_mosaics = self.patches['mosaic'].isin(testing_mosaics)
        
        # Use stratified splitting, i.e. each label is going to be split separately so we 
        # avoid having training and validation datasets which do not show the same distribution. However, 
        # we still want both datasets to be independent from each other. This means we need to split by polygons while 
        # keeping track of many patches we already have. 
        # For example, we want to split 85% training and 15% validation, i.e. 1000 patches are divided in 850 training
        # and 150 validation patches. But it might be that we have only 4 polygons in total (for simplicity we assume all of
        # them have the same number of patches). Hence, we have to split the patches into 75% training and 25% validation otherwise we would violate the principle of 
        # independency
        # 
        validation_polygons = []
        training_polygons = []
        testing_polygons = []

        for label, df in self.patches.groupby('label'):
            polygons = df.polygon.unique()
            r = np.random.RandomState(random_state)
            r.shuffle(polygons)
            if validation_split < 1:
                print('Using rational split')
                val_index = int(validation_split*len(polygons))
                validation_polygons = np.concatenate([validation_polygons, polygons[:val_index]])
                test_index = int(testing_split*len(polygons))
                testing_polygons = np.concatenate([testing_polygons, polygons[-test_index:]])
                training_polygons = np.concatenate([training_polygons, polygons[val_index:-test_index]])
            else:
                print('Using number split')
                validation_polygons = np.concatenate([validation_polygons, polygons[:validation_split]])
                testing_polygons = np.concatenate([testing_polygons, polygons[-testing_split:]])
                training_polygons = np.concatenate([training_polygons, polygons[validation_split:-testing_split]])
            
        self.training_patches = self.patches[self.patches.polygon.isin(training_polygons)]
        self.validation_patches = self.patches[self.patches.polygon.isin(validation_polygons)]
        self.testing_patches = self.patches[self.patches.polygon.isin(testing_polygons)]
        
        augmentator = iaa.SomeOf((0, None), [
            iaa.Add((-20, 20)),
            iaa.Crop(percent=(0, 0.02)),
            iaa.Affine(
                scale=(0.7, 1.3),
                rotate=(-20, 20), mode='reflect'),
            iaa.Fliplr(0.25), # horizontally flip 50% of the images
            iaa.Flipud(0.25),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0, 0.8)), # blur images with a sigma of 0 to 3.0
        ])
        
        self.training = ImageLoader(
            images=self.training_patches.index.values, 
            labels=self.training_patches.label.values,
            augmentator=augmentator, balance=balance_batch,
            preprocess_input=self.preprocess_input,
        )
        self.validation = ImageLoader(
            images=self.validation_patches.index.values, 
            labels=self.validation_patches.label.values,
            preprocess_input=self.preprocess_input,
        )
        self.testing = ImageLoader(
            images=self.testing_patches.index.values, 
            labels=self.testing_patches.label.values,
            preprocess_input=self.preprocess_input,
        )

        counts = dict(zip(
            *np.unique(self.training_patches.label.values, return_counts=True)
        ))
        self.class_weights = {
            i: max([(max(counts.values()) / counts[self.classes[i]]) / 2, 1])
            for i in range(len(self.classes))
        }
        self.class_indices = self.testing.class_indices

    def summary(self):
        summary = pd.DataFrame()
        summary['train'] = self.training_patches.label.value_counts()
        summary['train_ratio'] = \
            100*summary['train'] / summary['train'].sum()
        summary['valid'] = self.validation_patches.label.value_counts()
        summary['valid_ratio'] = \
            100*summary['valid'] / summary['valid'].sum()
        summary['test'] = self.testing_patches.label.value_counts()
        summary['test_ratio'] = \
            100*summary['test'] / summary['test'].sum()
        return summary
        
