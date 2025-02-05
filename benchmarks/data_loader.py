import pandas as pd
import json
import os
from src.data.custom_dataloader import CustomDataset
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
from PIL import Image
import numpy as np

def rgb_to_id(mask, id2rgb_dict):
    
    id2rgb = {tuple(v): int(k) for k, v in id2rgb_dict.items()}
    
    mask_array = np.array(mask)  
    id_mask = np.zeros(mask_array.shape[:2], dtype=np.uint8)  
    
    unique_classes = np.unique(mask_array.reshape(-1, mask_array.shape[-1]), axis=0)[1:]
    
    for rgb in unique_classes:
        rgb_tuple = tuple(rgb)  
        if rgb_tuple in id2rgb:  
            matches = np.all(mask_array == np.array(rgb), axis=-1).astype(np.uint8)
            id_mask[matches > 0] = id2rgb[rgb_tuple]
    
    return Image.fromarray(id_mask)

def get_pd_dataset(dataset_name, root, split="train"):
    json_file = "src/data/datasets_config.json"
    
    
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    data = json_data[dataset_name]
    image_key = data["image_key"]
    mask_key = data["label_key"]
    id2rgb_dict = data.get("id2rgb", {})
    
    df_path = f"benchmarks/dataframes/{dataset_name}_{split}.csv"
    
    # Load the dataset based on the type (HF or HF_disk)
    if data["type"] == "HF":
        dataset = load_dataset(data.get("hf_hub_path"), split=split, cache_dir=root)
    elif data["type"] == "HF_disk":
        dataset = load_from_disk(os.path.join(root, dataset_name, split))
    else:
        # Directly create dataframe with image and mask paths
        img_folder = os.path.join(root, dataset_name, split, image_key)
        mask_folder = os.path.join(root, dataset_name, split, mask_key)
        
        if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
            raise ValueError("Image folder or mask folder is missing.")
        
        image_names = sorted(os.listdir(img_folder))
        mask_names = sorted(os.listdir(mask_folder))
        
        records = []
        for img_name, mask_name in tqdm(zip(image_names, mask_names), 
                                total=len(image_names), 
                                desc="Processing images and masks", 
                                unit="file"):
            img_path = os.path.join(img_folder, img_name)
            mask_path = os.path.join(mask_folder, mask_name)
            
            mask = Image.open(mask_path).convert("RGB")
            id_mask = rgb_to_id(mask, id2rgb_dict)
            
            id_mask.save(mask_path, format="PNG")
            
            records.append({"image_paths": img_path, "mask_paths": mask_path})
        
        df = pd.DataFrame(records)
        df.to_csv(df_path, index=False)
        return df

    # Directory to store images and masks
    directory_path = os.path.join(root, "autogluon", dataset_name)
    os.makedirs(directory_path, exist_ok=True)

    # Process and save images and masks
    records = []
    for idx in tqdm(range(len(dataset)), total=len(dataset), desc="Processing dataset", unit="item"):
        img_path = os.path.join(directory_path, f"image_{idx}.jpg")
        mask_path = os.path.join(directory_path, f"mask_{idx}.png")
        
        img = dataset[idx][image_key]  # Assuming 'image' is a PIL image
        mask = dataset[idx][mask_key]  # Assuming 'mask' is a PIL image
        
        img.save(img_path, format="JPEG")
        mask.save(mask_path, format="PNG")
        
        records.append({"image_paths": img_path, "mask_paths": mask_path})
    
    df = pd.DataFrame(records)
    df.to_csv(df_path, index=False)
    
    return df

if __name__ == "__main__":
    dataset_names = ["sidewalk-semantic","semantic-drone-dataset", "FoodSeg103"]
    for dataset_name in dataset_names:
        df = get_pd_dataset(dataset_name, root = "/work/dlclarge2/dasb-Camvid/qtt_seg_datasets", split="train")
        df = get_pd_dataset(dataset_name, root = "/work/dlclarge2/dasb-Camvid/qtt_seg_datasets", split="test")
        print(f"{dataset_name} done!")