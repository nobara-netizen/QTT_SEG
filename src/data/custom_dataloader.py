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
from transformers import SamProcessor

def save_mask_with_prompts(mask, prompts, save_path="mask_with_prompts.png"):
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for (x, y) in prompts:
        cv2.circle(mask_vis, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(save_path, mask_vis)
    print(f"Saved: {save_path}")


def get_prompt(ground_truth_map, num_prompts):
    y_indices, x_indices = np.where(ground_truth_map > 0)
    foreground_points = list(zip(x_indices, y_indices)) 
    prompt = [list(random.choice(foreground_points)) for _ in range(num_prompts)]
    labels = [1 for _ in prompt]
    return prompt, labels


class CustomDataset():
    def __init__(
        self, 
        dataset_name,
        processor, 
        train=True, 
        resize_mask = True,
        multiclass = False,
        args=None
    ):
        self.args = args
        self.train = train
        self.dataset_name = dataset_name

        df_folder = "benchmarks/dataframes"

        if self.train:
            self.df = pd.read_csv(f"{df_folder}/{self.dataset_name}_train.csv")[:100]
        else:
            self.df = pd.read_csv(f"{df_folder}/{self.dataset_name}_test.csv")[:100]
        
        self.image_col = "image_paths"
        self.label_col = "mask_paths"

        self.processor = processor
        self.resize_mask = resize_mask
        self.multiclass = multiclass

    def apply_augmentations(self, image, mask, args):
        
        if args.horizontal_flip and random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        if args.vertical_flip and random.random() < 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        if args.random_rotate:
            angle = random.randint(-30, 30)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height))
            mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)

        return image, mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
            idx = int(idx)

            image_path = self.df[self.image_col].iloc[idx]
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if len(image.shape) == 3:  
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            else: 
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) 
            
            mask_path = self.df[self.label_col].iloc[idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if self.resize_mask:
                mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.uint8)
                prompts, labels = get_prompt(mask, 100)
                inputs = self.processor(
                    image,
                    input_points=[prompts],
                    input_labels = [labels], 
                    return_tensors="pt"
                )
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                inputs["ground_truth_mask"] = mask
            else:
                r = min(1024 / image.shape[1], 1024 / image.shape[0])
                new_width = int(image.shape[1] * r)
                new_height = int(image.shape[0] * r)

                image = cv2.resize(image, (new_width, new_height))
                mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.uint8)
                if self.train:
                    prompts, labels = get_prompt(mask,self.args.num_prompts)
                else:
                    prompts, labels = get_prompt(mask,self.args.num_prompts)
                inputs = {
                    "pixel_values" : image,
                    "ground_truth_mask" : mask,
                    "input_points" : np.array([prompts]),
                    "input_labels" : np.array([labels])
                }
            # save_mask_with_prompts((mask * 255).astype(np.uint8), prompts, save_path=f"prompt_visual_{idx}.png")
            return inputs
        # except Exception as e:
        #     print(f"Error occured: {e}")
        #     return {}

if __name__ == "__main__":
    dataset_name = "vineyard"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataset = CustomDataset(dataset_name, processor, train=False,resize_mask = True)
    inputs = dataset[1]
    for k, v in inputs.items():
        print(k, v.shape)
# ["eyes", "lesion", "tiktok", "fiber", "malignant","benign","building","FoodSeg103", "human_parsing_dataset", "sidewalk-semantic", "danish-golf-courses-orthophotos"]

# print([prompts])
# image_with_prompts = image.copy()
# for (x, y) in prompts:
#     cv2.circle(image_with_prompts, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # Red dots
# save_path = f"image_{idx}.png"
# cv2.imwrite(save_path, cv2.cvtColor(image_with_prompts, cv2.COLOR_RGB2BGR))

#   # get bounding box from mask
#     y_indices, x_indices = np.where(ground_truth_map > 0)
#     x_min, x_max = np.min(x_indices), np.max(x_indices)
#     y_min, y_max = np.min(y_indices), np.max(y_indices)
#     # add perturbation to bounding box coordinates
#     H, W = ground_truth_map.shape
#     x_min = max(0, x_min - np.random.randint(0, 20))
#     x_max = min(W, x_max + np.random.randint(0, 20))
#     y_min = max(0, y_min - np.random.randint(0, 20))
#     y_max = min(H, y_max + np.random.randint(0, 20))
#     bbox = [x_min, y_min, x_max, y_max]