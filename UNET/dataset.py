import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))
        image = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.] = 1.

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
