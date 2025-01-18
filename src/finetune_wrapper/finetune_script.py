from src.sam2_process import train
import argparse
import random
import os
import pandas as pd
import argparse
import json
import itertools
import logging

def finetune_script(
    job: dict,
    task_info : dict
):

    args = []

    config = job.get("config", {})
    config_id = job.get("config_id", None)
    fidelity = job.get("fidelity", 1)
    
    output_path = task_info.get("output_path", ".")
    output_dir = os.path.join(output_path, str(config_id))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    task_info.pop("output_path")
    
    # add job info items
    args.extend([
        "--num_train_epochs", str(fidelity),
        "--output_dir" , output_dir
    ])

    # add task info items
    for key, value in task_info.items():
        args.extend([f"--{key}", str(value)])

    # add config items
    for key, value in config.items():
        args.extend([f"--{key}", str(value)])

    # finetune based on model type
    model_name = config.get("model_name", "")
    return_scores_per_epoch = task_info.get("return_scores_per_epoch", False)
 

    if model_name == "SAM":
        parser = train.get_parser()
        args, _ = parser.parse_known_args(args)
        
        if return_scores_per_epoch:
            score,cost, scores_per_epoch = train.main(args)
        else:
            score,cost = train.main(args)

    else:
        raise ValueError("Model Type not supported yet! Please try another type.")
    
    report = job.copy()
    report["score"] = score
    report["cost"] = cost
    report["status"] = True  
    report["info"] = {"path": output_dir}

    if return_scores_per_epoch:
        return report, scores_per_epoch

    return report

    

if __name__ == "__main__":
    
    num_configs = 1
    fidelity = 50
    output_path = "outputs"
    train_csv_path = "src/finetune_wrapper/finetuning_results.csv"
    return_scores_per_epoch = True

    # Define the hyperparameter grid
    hyperparameter_grid = {
        "learning_rate": [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        "weight_decay": [0, 1e-05, 0.0001, 0.001, 0.01, 0.1],
        "model_name": ["SAM"],
        "lora" : [0],
        "lora_targets": ["image_enc_attn", "image_enc_mlp", "memory_self_attn", "memory_cross_attn"],
        "lora_rank": [2, 4, 8],
        "lora_alpha": [8, 16, 32],
        "lora_dropout": [0.0, 0.1, 0.2],
        "horizontal_flip": [0,1],
        "vertical_flip": [0,1],
        "random_rotate": [0,1],
        "elastic_transform": [0,1],
        "normalize": [0,1],
        "opt": ["sgd", "adam", "adamw"],
        "momentum": [0.8, 0.9, 0.99],
        "opt_betas": [(0.9, 0.999), (0.85, 0.995)],
        "sched": ["cosine", "step", "multistep", "plateau"],
        "decay_epochs": [10, 20, 30],
        "decay_rate": [0.001, 0.01, 0.1],
    }

    random_combinations = [
        {key: random.choice(value) for key, value in hyperparameter_grid.items()}
        for _ in range(num_configs)
    ]

    base_columns = list(hyperparameter_grid.keys()) + ["cost", "score"]
    epoch_columns = [f"epoch_{i}" for i in range(fidelity)]
    all_columns = base_columns + epoch_columns

    if not os.path.exists(train_csv_path):
        pd.DataFrame(columns=all_columns).to_csv(train_csv_path, index=False)

    for i, combination_dict in enumerate(random_combinations):
        dataset_name = random.choice(["sidewalk-semantic", "human_parsing_dataset", "danish-golf-courses-orthophotos", "semantic-drone-dataset", "FoodSeg103"])
        
        
        job = {
            "config" : combination_dict,
            "config_id" : i,
            "fidelity" : fidelity
        }

        task_info = {
        "dataset_name" : dataset_name,
        "output_path" : output_path,
        "return_scores_per_epoch" : return_scores_per_epoch 
        }

        if return_scores_per_epoch:
            report, scores_per_epoch = finetune_script(job, task_info)
            result = combination_dict.copy()
            result["cost"] = report["cost"]
            result["score"] = report["score"]
            result.update(scores_per_epoch)
        
        else:
            report = finetune_script(job, task_info)
            result = combination_dict.copy()
            result["cost"] = report["cost"]
            result["score"] = report["score"]

        
        pd.DataFrame([result]).to_csv(train_csv_path, mode='a', index=False, header=False)
    
    df = pd.read_csv(train_csv_path)
    print(df["score"].head())