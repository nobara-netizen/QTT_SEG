import uuid
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import os
import time

from qtt import QuickTuner, QuickOptimizer
from qtt.predictors import PerfPredictor, CostPredictor
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    OrConjunction,
    OrdinalHyperparameter,
)
from src.finetune_wrapper.finetune_script import finetune_script, get_meta_data
from src.sam2_process.test import test
from src.data.custom_dataloader import CustomDataset

output_path = "/work/dlclarge2/dasb-Camvid/config_checkpoints"

all_columns = ["DATASET","QTT_CONFIG", "QTT_TUNING_SCORE", "TIME_BUDGET","QTT_TEST_DICE", "ZERO_SHOT_DICE","TEST IMPROVEMENT"]
time_budgets = [60, 120, 240, 300, 600]
dataset_names = ["building", "lesion", "FoodSeg103", "semantic-drone-dataset", "human_parsing_dataset", "sidewalk-semantic", "danish-golf-courses-orthophotos"]
seeds = [0, 1, 7, 42, 99, 123, 256, 512, 1024, 1337, 2024, 31415, 54321, 65536, 99999]
results_csv_path = "/home/dasb/workspace/QTT_SEG/results.csv"

def delete_folders():
    print("Flushing stale files...")
    folder_paths = [
        Path("/home/dasb/workspace/QTT_SEG/qtt"),
        Path(f"/work/dlclarge2/dasb-Camvid/qtt_states"),
        Path(output_path)  
    ]
    for folder_path in folder_paths:
        if folder_path.exists():  
            subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
            for folder in subfolders:
                try:
                    print(f"Deleting: {folder}")
                    shutil.rmtree(folder)  
                except Exception as e:
                    print(f"Failed to delete {folder}: {e}")
                    continue
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    OrConjunction,
    OrdinalHyperparameter,
)

def get_config_space():
    cs = ConfigurationSpace("cv-segmentation")

    # Learning rate and weight decay
    lr = OrdinalHyperparameter("learning_rate", [0.0001, 0.00001, 0.000001])  # Matches allowed choices
    wd = OrdinalHyperparameter("weight_decay", [1e-4, 1e-5])  # Matches allowed choices

    # Model selection
    model = Categorical("model_name", ["SAM"])

    # LoRA hyperparameters
    lora = OrdinalHyperparameter("lora", [0])  # Matches get_parser() default
    lora_targets = Categorical("lora_targets", ["image_enc_attn", "image_enc_mlp", "memory_self_attn", "memory_cross_attn"])
    lora_rank = OrdinalHyperparameter("lora_rank", [4, 8])  # Matches choices
    lora_dropout = OrdinalHyperparameter("lora_dropout", [0.1, 0.2, 0.3])  # Matches choices

    # Augmentation hyperparameters
    horizontal_flip = OrdinalHyperparameter("horizontal_flip", [0, 1])
    vertical_flip = OrdinalHyperparameter("vertical_flip", [0, 1])
    random_rotate = OrdinalHyperparameter("random_rotate", [0, 1])
    elastic_transform = OrdinalHyperparameter("elastic_transform", [0, 1])
    normalize = OrdinalHyperparameter("normalize", [0, 1])

    # Optimizer arguments
    opt = Categorical("opt", ["sgd", "adam", "adamw", "rmsprop"])  # Added "rmsprop" to match parser
    momentum = OrdinalHyperparameter("momentum", [0.8, 0.9])  # Matches allowed values
    opt_betas = Categorical("opt_betas", [(0.9, 0.999), (0.85, 0.995), (0.8, 0.9)])  # Matches choices

    # Scheduler arguments
    sched = Categorical("sched", ["cosine", "onecycle", "plateau"])
    decay_epochs = OrdinalHyperparameter("decay_epochs", [30, 60, 90])  # Matches allowed choices
    decay_rate = OrdinalHyperparameter("decay_rate", [0.1, 0.3, 0.5])  # Matches allowed choices
    patience_epochs = OrdinalHyperparameter("patience_epochs", [5])  # Added since it's part of sched args

    # Adding hyperparameters to configuration space
    cs.add_hyperparameters([
        lr, wd, model, lora, lora_targets, lora_rank, lora_dropout,
        horizontal_flip, vertical_flip, random_rotate, elastic_transform, normalize,
        opt, momentum, opt_betas, sched, decay_epochs, decay_rate, patience_epochs
    ])

    return cs


if __name__ == "__main__":
    
    train_predictors = False
    perf_predictor_path = "/home/dasb/.cache/qtt_new/PerfPredictor/"
    cost_predictor_path = "/home/dasb/.cache/qtt_new/CostPredictor/"

    meta_df = pd.read_csv("/home/dasb/workspace/QTT_SEG/src/finetune_wrapper/finetuning_results.csv")

    cost = meta_df["cost"]
    curve = meta_df.filter(regex=r'^epoch_\d{1,2}_iou$')
    config = meta_df.drop(columns=["cost", "dataset", "opt_betas"] + curve.columns.tolist())
    X = config
    y = curve.values

    fit_params = {
        "batch_size": 100,
    }
    if train_predictors:
        perf_predictor = PerfPredictor(fit_params).fit(X, y)
        perf_predictor.save(perf_predictor_path)
        y = cost.values.reshape(-1,1)
        cost_predictor = CostPredictor(fit_params).fit(X, y)
        cost_predictor.save(cost_predictor_path)

    else:
        perf_predictor = PerfPredictor().load(perf_predictor_path)
        cost_predictor = CostPredictor().load(cost_predictor_path)

    for i in range(30): 
        seed = seeds[i % len(seeds)]
        for time_budget in time_budgets:
            for dataset_name in dataset_names:
                delete_folders()
                cs = get_config_space()
                optimizer = QuickOptimizer(
                    cs,
                    max_fidelity=50,
                    perf_predictor=perf_predictor,
                    cost_predictor=cost_predictor,
                    cost_aware=True,
                    cost_factor=1.0,
                    acq_fn="ei",
                    patience=5,
                    tol=0.001,
                    refit=False,
                    path=f"/work/dlclarge2/dasb-Camvid/qtt_states/{uuid.uuid4().hex}",
                    seed=seed  
                )

                task_info = {
                    "dataset_name": dataset_name,
                    "output_path": output_path,
                    "seed" : seed
                }

                metafeat = get_meta_data(dataset_name)

                print("Optimiser Setup")
                optimizer.setup(100, metafeat)

                tuner = QuickTuner(
                    optimizer,
                    finetune_script,
                )

                traj, runtime, history = tuner.run(fevals=100, trial_info=task_info, time_budget=time_budget)
                config_id, config, score, budget, cost, info = tuner.get_incumbent()

                print("\n***** PERFORMANCE COMPARISON QTT-SEG *****")
                print("=================================")

                zero_shot_iou = test(
                    dataset_name=dataset_name,
                    zero_shot=True,
                    args=config
                )

                qtt_iou = test(
                    dataset_name=dataset_name,
                    zero_shot=False,
                    predicted_model_path=f"{info['path']}/sam2model.torch",
                    args=config
                )

                iou_imp = ((qtt_iou - zero_shot_iou) / zero_shot_iou) * 100
                print("IOU Improvement: {:.2f}%".format(iou_imp))
                print("---------------------------------")

                result = {
                    "DATASET": dataset_name,
                    "SEED" : seed,
                    "QTT_CONFIG": config,
                    "QTT_TUNING_SCORE": score,
                    "TIME_BUDGET": time_budget,
                    "QTT_TEST_IOU": qtt_iou,
                    "ZERO_SHOT_IOU": zero_shot_iou,
                    "TEST IMPROVEMENT": iou_imp
                }
                if os.path.exists(results_csv_path):
                    pd.DataFrame([result]).to_csv(results_csv_path, mode='a', index=False, header=False)
                else:
                    pd.DataFrame([result]).to_csv(results_csv_path, index=False)
