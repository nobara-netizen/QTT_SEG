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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from sklearn.model_selection import train_test_split


def save_mask_with_prompts(mask, prompts, save_path="mask_with_prompts.png"):
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (x, y) in prompts:
        cv2.circle(mask_vis, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(save_path, mask_vis)
    print(f"Saved: {save_path}")

def save_mask_with_bbox(mask, bbox, save_path="mask_with_bbox.png"):

    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(mask_vis, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    cv2.imwrite(save_path, mask_vis)
    print(f"Saved: {save_path}")

def get_prompt(ground_truth_map, num_prompts):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    foreground_points = list(zip(x_indices, y_indices)) 
    prompt = [list(random.choice(foreground_points)) for _ in range(num_prompts)]
    labels = [1 for _ in prompt]
    return prompt, labels

def get_bounding_box_prompt(ground_truth_map):
    y_indices, x_indices = np.where(ground_truth_map > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None, None  

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    bounding_box = [x_min, y_min, x_max, y_max]

    return bounding_box


def get_random_bounding_box(ground_truth_map, min_area=50):
    binary_mask = (ground_truth_map > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, x + w, y + h])

    if not boxes:
        return None
    return random.choice(boxes)

def get_largest_bounding_box(ground_truth_map, min_area=50):
    """
    Returns the bounding box of the largest connected component in the binary mask.
    
    Args:
        ground_truth_map (np.ndarray): 2D binary mask.
        min_area (int): Ignore very small regions (noise).
        
    Returns:
        bounding_box (list): [x_min, y_min, x_max, y_max] or None if no object.
    """
    binary_mask = (ground_truth_map > 0).astype(np.uint8)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_box = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area and area > max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            best_box = [x, y, x + w, y + h]
            max_area = area

    return best_box

class CustomDataset():
    def __init__(self, dataset_name, split="test", multiclass=False, args=None):
        self.args = args
        self.split = split
        self.dataset_name = dataset_name
        self.multiclass = multiclass

        df_folder = "benchmarks/dataframes"
        full_df = pd.read_csv(f"{df_folder}/{self.dataset_name}_train.csv")[:100]
        train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
        test_df = pd.read_csv(f"{df_folder}/{self.dataset_name}_test.csv")[:100]

        if self.split == "train":
            self.df = train_df
        elif self.split == "val":
            self.df = val_df
        else:
            self.df = test_df

        self.image_col = "image_paths"
        self.label_col = "mask_paths"

        if self.split == "train" and self.args:
            transforms = [
                A.HorizontalFlip(p=0.5) if self.args.horizontal_flip else None,
                A.VerticalFlip(p=0.5) if self.args.vertical_flip else None,
                A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5) if self.args.random_rotate else None,                
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ]
            self.transform = A.Compose(
                [t for t in transforms if t is not None],
                additional_targets={'mask': 'mask'}
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        image_path = self.df[self.image_col].iloc[idx]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            print(f"[Warning] Empty or invalid image at {image_path}. Skipping index {idx}.")
            return self.__getitem__((idx + 1) % len(self))  

        if len(image.shape) == 3:  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        else: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 

        mask_path = self.df[self.label_col].iloc[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None or mask.size == 0 or np.sum(mask) == 0:
            print(f"[Warning] Empty or invalid mask at {mask_path}. Skipping index {idx}.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        r = min(1024 / image.shape[1], 1024 / image.shape[0])
        new_width = int(image.shape[1] * r)
        new_height = int(image.shape[0] * r)

        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)
        prompt = get_bounding_box_prompt(mask)

        inputs = {
            "pixel_values" : image,
            "ground_truth_mask" : mask,
            "input_box" : np.array(prompt).reshape(1,4)
        }
        # save_mask_with_bbox((mask * 255).astype(np.uint8), prompt, save_path=f"prompt_visual_{idx}.png")
        return inputs


if __name__ == "__main__":
    dataset_name = "leaf"
    dataset = CustomDataset(dataset_name, split="train")
    inputs = dataset[1]
    for k, v in inputs.items():
        print(k, v.shape)
