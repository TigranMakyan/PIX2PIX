import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import config

class MapDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]    # replace 600 bay half of image width
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=image, image0=target_image)
        input_image, target_image = augmentations['image'], augmentations['image0']

        input_image = config.transform_only_input(image=input_image)['image']
        target_image = config.transform_only_input(image=target_image)['image']

        return input_image, target_image
