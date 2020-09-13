import os
import cv2
from torch.utils.data import Dataset

import simulation


class SimDataset(Dataset):
    def __init__(self, count, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.target_masks[idx]
        if self.transform:
            image = self.transform(image)

        return [image, mask]

class CustomDataset(Dataset):

    def __init__(self, count, image_path, mask_path=None, transform=None):
        self.input_images, self.target_masks = simulation.generate_random_data(192, 192, count=count)
        self.transform = transform

        image_names = os.listdir(image_path)
        self.image_names = [os.path.join(image_path, image_name) for image_name in image_names if image_name.endswith(('jpg', 'png'))]

        self.mask_names = None

        if mask_path:
            mask_names = os.listdir(mask_path)
            self.mask_names = [os.path.join(mask_path, mask_name) for mask_name in mask_names if mask_name.endswith(('jpg', 'png'))]

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        mask_name = self.mask_names[idx] if self.mask_names else None

        image = cv2.imread(image_name)
        mask = cv2.imread(mask_name) if mask_name else None

        if self.transform:
            image = self.transform(image)

        return image, mask

