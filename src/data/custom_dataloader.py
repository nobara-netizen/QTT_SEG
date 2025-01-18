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
    def __init__(self, dataset_name, root, train=True, args=None):
        
        self.args = args
        self.train = train
        
        json_file = "src/data/datasets_config.json"
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        self.dataset_name = dataset_name
        data = json_data[self.dataset_name]
        
        self.hf = False
        if self.train:
            print("Loading Train Set!")
            self.split = data.get("train_split")
        else:
            print("Loading Test Set!")
            self.split = data.get("test_split")
        
        self.image_key = data.get("image_key")
        self.label_key = data.get("label_key")
        
        
        if data["type"] == "HF":
            self.dataset = load_dataset(data.get("hf_hub_path"), split=self.split, cache_dir = root)
            self.hf = True
        
        elif data["type"] == "HF_disk":
            self.dataset = load_from_disk(os.path.join(root, self.dataset_name, self.split))
            self.hf = True

        else:
            self.img_folder = os.path.join(root, self.dataset_name, self.split , self.image_key)
            self.mask_folder = os.path.join(root, self.dataset_name, self.split , self.label_key)
            
            # Ensure both folders exist before reading
            if self.img_folder and self.mask_folder:
                self.image_names = sorted(os.listdir(self.img_folder))  
                self.mask_names = sorted(os.listdir(self.mask_folder)) 
            else:
                raise ValueError("Image folder or mask folder is missing.")
        
        # Set `id2label` and `id2rgb` with default None if not found
        self.id2label = data.get("id2label", None)
        self.id2rgb = data.get("id2rgb", None)


    def apply_augmentations(self,image, mask, args):
        if args == None:
            return image, mask
        """
        Apply augmentations to the image and mask.

        Args:
        - image (numpy array): The input image.
        - mask (numpy array): The corresponding mask.
        - args (Namespace): The augmentation arguments containing various flags.

        Returns:
        - image (numpy array): The transformed image.
        - mask (numpy array): The transformed mask.
        """

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
            angle = random.randint(-30, 30)  # Random angle between -30 and 30 degrees
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height))
            mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)

        # Random Crop
        if args.random_crop_size:
            crop_size = ast.literal_eval(args.random_crop_size)
            image, mask = self.random_crop(image, mask, crop_size)

        # Normalize
        if args.normalize:
            image = (image / 255.0 - 0.5) / 0.5  # Example normalization to [0, 1] range
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        return image, mask


    def random_crop(self,image, mask, crop_size):
        """
        Apply random crop to the image and mask.

        Args:
        - image (numpy array): The input image.
        - mask (numpy array): The corresponding mask.
        - crop_size (tuple): The desired crop size (height, width).

        Returns:
        - cropped_image (numpy array): The cropped image.
        - cropped_mask (numpy array): The cropped mask.
        """
        height, width = image.shape[:2]
        crop_height, crop_width = crop_size

        if crop_height > height or crop_width > width:
            raise ValueError("Crop size must be smaller than image size")

        # Randomly choose the top-left corner of the crop
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)

        cropped_image = image[top:top + crop_height, left:left + crop_width]
        cropped_mask = mask[top:top + crop_height, left:left + crop_width]

        return cropped_image, cropped_mask

    def __len__(self):
        if self.hf:
            return len(self.dataset)
        else:
            return len(self.image_names)

    def __getitem__(self, idx):

        if self.hf:
            item = self.dataset[idx]
            image = np.array(item[self.image_key])
            mask = np.array(item[self.label_key])

        else:

            img_path = os.path.join(self.img_folder, self.image_names[idx])  
            mask_path = os.path.join(self.mask_folder, self.mask_names[idx])  
            
            image = np.array(Image.open(img_path))
            mask = np.array(Image.open(mask_path))

        if self.train:
            image, mask = self.apply_augmentations(image, mask, self.args)
        
        self.resize_factor = np.min([1024 / image.shape[1], 1024 / image.shape[0]])        
        
       
        image = cv2.resize(image, (int(image.shape[1] * self.resize_factor), int(image.shape[0] * self.resize_factor)))
        mask = cv2.resize(mask, (int(mask.shape[1] * self.resize_factor), int(mask.shape[0] * self.resize_factor)), interpolation=cv2.INTER_NEAREST)
       

        binary_masks = []
        points = [] 

        if len(mask.shape) == 2:
            unique_classes = np.unique(mask)[1:]  
        else:
            unique_classes = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0)[1:]
        
        if self.id2label:
            assert isinstance(self.id2label, dict), "The variable id2label is not a dictionary!" 
            
            self.id2label = {int(k): v for k, v in self.id2label.items()}
            for index in unique_classes:
                
                binary_mask = (mask == index).astype(np.uint8) 
                binary_masks.append(binary_mask)
                coords = np.argwhere(binary_mask > 0) 
               
                if len(coords) > 0:
                    yx = coords[np.random.randint(len(coords))]
                    points.append([[yx[1], yx[0]]])  
                    
            
            if len(unique_classes) != len(points):
                raise ValueError(f'Total unique classes: {len(unique_classes)} is not equal to points: {len(points)}')
        
        elif self.id2rgb: 
            assert isinstance(self.id2rgb, dict), "The variable id2rgb is not a dictionary!" 
            
            self.id2rgb = {int(k): v for k, v in self.id2rgb.items()}
            for rgb_value in unique_classes:
                binary_mask = np.all(mask == np.array(rgb_value), axis=-1).astype(np.uint8)
                binary_masks.append(binary_mask)
                coords = np.argwhere(binary_mask > 0) 
                if len(coords) > 0:
                    yx = coords[np.random.randint(len(coords))]
                    points.append([[yx[1], yx[0]]])
            
            if len(unique_classes) != len(points):
                raise ValueError(f'Total unique classes: {len(unique_classes)} is not equal to points: {len(points)}')
        
        else:
            raise ValueError("Either id2label or id2rgb dict is required incase of multiclass!")

        
        return image, np.array(binary_masks), np.array(points), np.ones([len(points), 1]) 



if __name__ == "__main__":
    dataset_name = "scene_parse_150"
    root = "/work/dlclarge2/dasb-Camvid/qtt_seg_datasets"
    dataset = CustomDataset(dataset_name, root, train=True)
    print(dataset[0])