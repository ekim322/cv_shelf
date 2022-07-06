import glob
import os
from PIL import Image
import numpy as np
from cv2 import resize
import albumentations as A

import torch
from torch.utils.data import Dataset

class Triplet_Dataset(Dataset):
    def __init__(self, img_dir, img_size=128, transform=None):
        img_dir_tmp = os.path.join(img_dir, '**/*.jpg')
        img_paths = glob.glob(img_dir_tmp, recursive=True)
        img_classes_str = [os.path.basename(os.path.dirname(x)) for x in img_paths]
        unique_classes = np.sort(np.unique(img_classes_str))
        class_keys = {unique_classes[i]: range(len(unique_classes))[i]
                      for i in range(len(unique_classes))}
        img_classes = np.array([class_keys[img_classes_str[i]] for i in range(len(img_classes_str))])

        transform = A.Compose([
            A.RandomCrop(width=int(img_size*0.9), height=int(img_size*0.9)),
            A.Resize(width=img_size, height=img_size),
            A.RandomRotate90()
        ])

        self.img_size = img_size
        self.transform = transform
        self.img_paths = img_paths
        self.img_classes = img_classes
        self.unique_classes = np.unique(img_classes)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        anc_img_path = self.img_paths[idx]
        anc_img_class = self.img_classes[idx]

        pos_possible = np.where(np.array(self.img_classes)==anc_img_class)[0]
        pos_possible = np.delete(pos_possible, np.where(pos_possible==idx))
        pos_img_idx = np.random.choice(pos_possible)
        pos_img_path = self.img_paths[pos_img_idx]

        neg_possible = np.where(np.array(self.img_classes)!=anc_img_class)[0]
        neg_img_idx = np.random.choice(neg_possible)
        neg_img_path = self.img_paths[neg_img_idx]
        neg_img_class = self.img_classes[neg_img_idx]

        anc_img = np.array(Image.open(anc_img_path).convert("RGB"))
        pos_img = np.array(Image.open(pos_img_path).convert("RGB"))
        neg_img = np.array(Image.open(neg_img_path).convert("RGB"))

        if self.transform is not None:
            anc_img = self.transform(image=anc_img)["image"]
            pos_img = self.transform(image=pos_img)["image"]
            neg_img = self.transform(image=neg_img)["image"]

        anc_img = (anc_img / np.max(anc_img)).astype(np.float32)
        pos_img = (pos_img / np.max(pos_img)).astype(np.float32)
        neg_img = (neg_img / np.max(neg_img)).astype(np.float32)

        target_dict = {
            'anc_img': torch.from_numpy(np.transpose(anc_img, (2, 0, 1))),
            'pos_img': torch.from_numpy(np.transpose(pos_img, (2, 0, 1))),
            'neg_img': torch.from_numpy(np.transpose(neg_img, (2, 0, 1))),
            'pos_label': anc_img_class,
            'neg_label': neg_img_class,
        }

        return target_dict