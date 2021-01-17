from dataset import read_xray_image, save_array_to_image
from config import cfg
import os

train_dir = cfg.DATASET.TRAIN_DIR
i = 1
for dirname, _, filenames in os.walk(train_dir):
    for filename in filenames:
        print(f"{i:05d}/{len(filenames)}", end='\r')
        # read dicom data
        # read dicom data
        img_arry, sex = read_xray_image(os.path.join(dirname, filename))
        # save array
        save_array_to_image(img_arry, 'new_data/'+filename[:-6])
        i = i+1