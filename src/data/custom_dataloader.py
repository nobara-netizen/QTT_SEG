from kagglehub import dataset_download
from huggingface_hub import hf_hub_download

from datasets import load_from_disk, load_dataset
import numpy as np
import os
from pathlib import Path
import json
from PIL import Image
import cv2
import pandas as pd
import ast
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, Resize, Normalize, RandomCrop, ElasticTransform, Compose
)
from albumentations.pytorch import ToTensorV2
import random

class CustomDataset():
    def __init__(self, dataset_name, train=True, args=None, sam1=False):
        self.args = args
        self.train = train
        self.dataset_name = dataset_name
        self.sam1 = sam1
        df_folder = "benchmarks/dataframes"
        if self.train:
            self.df = pd.read_csv(f"{df_folder}/{self.dataset_name}_train.csv")
        else:
            self.df = pd.read_csv(f"{df_folder}/{self.dataset_name}_test.csv")
        
        self.image_col = "image_paths"
        self.label_col = "mask_paths"

    def apply_augmentations(self, image, mask, args):
        if args is None:
            return image, mask
        # Horizontal Flip
        if args.horizontal_flip and random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        # Vertical Flip
        if args.vertical_flip and random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        # Random Rotation
        if args.random_rotate:
            angle = random.randint(-30, 30)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height))
            mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        # Normalize
        if args.normalize:
            image = (image / 255.0 - 0.5) / 0.5
        
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        return image, mask

    def random_crop(self, image, mask, crop_size):
        height, width = image.shape[:2]
        crop_height, crop_width = crop_size
        if crop_height > height or crop_width > width:
            raise ValueError("Crop size must be smaller than image size")
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        cropped_image = image[top:top + crop_height, left:left + crop_width]
        cropped_mask = mask[top:top + crop_height, left:left + crop_width]
        return cropped_image, cropped_mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        image = np.array(Image.open(self.df[self.image_col].iloc[idx]))
        mask = np.array(Image.open(self.df[self.label_col].iloc[idx]))
        if self.train:
            image, mask = self.apply_augmentations(image, mask, self.args)
        if self.sam1:
            max_dim = 256
            image = cv2.resize(image, (max_dim, max_dim))
            mask = cv2.resize(mask, (max_dim, max_dim), interpolation=cv2.INTER_NEAREST)
        else:
            max_dim = 1024
            self.resize_factor = np.min([max_dim / image.shape[1], 
                                        max_dim / image.shape[0]])
            image = cv2.resize(image, (int(image.shape[1] * self.resize_factor), int(image.shape[0] * self.resize_factor)))
            mask = cv2.resize(mask, (int(mask.shape[1] * self.resize_factor), int(mask.shape[0] * self.resize_factor)), interpolation=cv2.INTER_NEAREST)

        binary_masks = []
        points = [] 
        unique_classes = np.unique(mask)
        for index in unique_classes:
            binary_mask = (mask == index).astype(np.uint8)
            binary_masks.append(binary_mask)
            coords = np.argwhere(binary_mask > 0)
            if len(coords) > 0:
                yx = coords[np.random.randint(len(coords))]
                points.append([[yx[1], yx[0]]])
        if len(unique_classes) != len(points):
            raise ValueError(f'Total unique classes: {len(unique_classes)} is not equal to points: {len(points)}')
        if self.sam1:
            return {"image": image, "label":np.array(binary_masks)}
        return image, np.array(binary_masks), np.array(points), np.ones([len(points), 1])

if __name__ == "__main__":
    dataset_name = "landslides"
    dataset = CustomDataset(dataset_name, train=False)
    img, masks, points, labels = dataset[0]
    print(img.shape, masks.shape, points.shape, labels.shape)