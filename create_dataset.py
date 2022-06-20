import os, cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset

class CustomDataset(Dataset):
    def __init__(self, images_dir, masks_dir, class_rgb_values=None, augmentation=None, preprocessing=None, ):

        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]

        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):

        # print(self.mask_paths[i])
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[i], cv2.IMREAD_UNCHANGED)[:, :, 3]
        # print(mask)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        # visualize(image=image,gt_mask=mask, binar_mask=cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1])

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            mask = mask.unsqueeze(0)
            mask = mask / 255.0


        else:
            image = np.rollaxis(image, 2, 0)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=0)

        return torch.Tensor(image), torch.Tensor(mask)