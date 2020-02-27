"""Create fold files for training and test images
"""


from typhon.files import FileSet
from sklearn.model_selection import train_test_split

df = pd.concat([
    FileSet('/scratch-a/jmrziglod/sen2agri/data/malawi_summer/patches/original/{label}/*.png').to_dataframe(),
    FileSet('/scratch-a/jmrziglod/sen2agri/data/malawi_summer/patches/augmented/{label}/*.png').to_dataframe()
])

train_files, test_files, _, _ = train_test_split(df.index.values, df.label.values, shuffle=True, test_size=0.1)

with open('/home/jmrziglod/projects/sen2agri/drone-crop-type/folds/malawi_summer/train_test_all_mosaics/train.txt', 'w') as txt_file:
    txt_file.write("\n".join(train_files))
with open('/home/jmrziglod/projects/sen2agri/drone-crop-type/folds/malawi_summer/train_test_all_mosaics/test.txt', 'w') as txt_file:
    txt_file.write("\n".join(test_files))