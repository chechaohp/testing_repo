from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2

class VinDataset(Dataset):
    
    def __init__(self, cfg, transforms=None):
        super().__init__()
        # Load csv
        ROOT_DIR = cfg.DATASET.ROOT
        CSV_FILE = cfg.DATASET.CSV
        TRAIN_DIR = cfg.DATASET.TRAIN_DIR

        self.df = pd.read_csv(f'{ROOT_DIR}/{CSV_FILE}')
        self.image_ids = self.df["image_id"].unique()
        self.image_dir = f'{ROOT_DIR}/{TRAIN_DIR}' 
        self.transforms = transforms
        
    def __getitem__(self, index):
        
        image_id = self.image_ids[index]
        records = self.df[(self.df['image_id'] == image_id)]
        records = records.reset_index(drop=True)

        # dicom = pydicom.dcmread(f"{self.image_dir}/{image_id}.dicom")
        image = cv2.imread(f"{self.image_dir}/{image_id}.png")
       
        if records.loc[0, "class_id"] == 0:
            records = records.loc[[0], :]
        
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        labels = torch.tensor(records["class_id"].values, dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.tensor(sample['bboxes'])

        if target["boxes"].shape[0] == 0:
            # Albumentation cuts the target (class 14, 1x1px in the corner)
            target["boxes"] = torch.from_numpy(np.array([[0.0, 0.0, 1.0, 1.0]]))
            target["area"] = torch.tensor([1.0], dtype=torch.float32)
            target["labels"] = torch.tensor([0], dtype=torch.int64)
            
        return image, target
    
    def __len__(self):
        return self.image_ids.shape[0]