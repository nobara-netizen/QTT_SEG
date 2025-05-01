from src.sam2_process import sam2_train
from src.sam1_process import sam1_train
import argparse
import random
import os
import pandas as pd
import argparse
import json
import itertools
import logging
import numpy as np
from PIL import Image
import random
from ConfigSpace import Configuration
from src.utils.utils import get_config_space
import torch

def finetune_script(
    job: dict,
    trial_info : dict
):
    args = []

    config = job.get("config")
    config_id = job.get("config-id")
    fidelity = job.get("fidelity")
    
    output_path = trial_info.get("output_path", ".")
    output_dir = os.path.join(output_path, str(config_id))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trial_info.pop("output_path")
    
    # add job info items
    args.extend([
        "--num_train_epochs", str(fidelity),
        "--output_dir" , output_dir
    ])

    # add task info items
    for key, value in trial_info.items():
        args.extend([f"--{key}", str(value)])

    # add config items
    for key, value in config.items():
        args.extend([f"--{key}", str(value)])

    return_scores_per_epoch = trial_info.get("return_scores_per_epoch", False)
 
    parser = sam2_train.get_parser()
    args, _ = parser.parse_known_args(args)
    if return_scores_per_epoch:
        result, ious = sam2_train.main(args)
    else:
        result = sam2_train.main(args)

    report = job.copy()
    report.update(result)
    report["status"] = True  
    report["info"] = {"path": output_dir}

    if return_scores_per_epoch:
        return report, ious

    return report

def get_meta_data(dataset_name):
    df_folder = "benchmarks/dataframes"
    df = pd.read_csv(f"{df_folder}/{dataset_name}_train.csv")
    
    num_samples = len(df)
    
    image_col = "image_paths"
    label_col = "mask_paths"
    
    img = np.array(Image.open(df[image_col].iloc[0]))
    mask = np.array(Image.open(df[label_col].iloc[0]))
    num_classes = len(np.unique(mask)[1:])

    if len(img.shape) > 2:
        num_channels = 3
    else:
        num_channels = 1
    
    return {
        "num_samples": num_samples,
        "default_resolution": (img.shape[0], img.shape[1]),
        "num_channels": num_channels,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    num_configs = 2000
    fidelity = 10
    output_path = "outputs"
    train_csv_path = "src/finetune_wrapper/finetuning_results_sam.csv"
    return_scores_per_epoch = True

    cs = get_config_space()
    i = 0
    while i <  num_configs:
        try:
            dataset_name = np.random.choice(["leaf", "polyp", "eyes", "lesion", "fiber", "building"])
            print(dataset_name)
            config = cs.sample_configuration()
            job = {
                "config" : config,
                "config-id" : i,
                "fidelity" : fidelity
            }

            trial_info = {
            "dataset_name" : dataset_name,
            "output_path" : output_path,
            "return_scores_per_epoch" : return_scores_per_epoch 
            }

            meta = get_meta_data(dataset_name)

            if return_scores_per_epoch:
                report, scores_per_epoch = finetune_script(job, trial_info)
                config = config.get_dictionary()
                dataset = report["dataset"]
                cost = report["cost"]
            
            else:
                report = finetune_script(job, trial_info)

            data = {**config, **meta, 'dataset': dataset, 'cost': cost, **scores_per_epoch}

            if os.path.exists(train_csv_path):
                pd.DataFrame([data]).to_csv(train_csv_path, mode='a', index=False, header=False)
            else:
                pd.DataFrame([data]).to_csv(train_csv_path, index=False)
            i += 1
        except Exception as e:
            print(f"Error occured: {e}")
            continue

