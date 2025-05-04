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
from src.sam2_process.sam2_test import test
from src.data.custom_dataloader import CustomDataset
from src.utils.utils import get_config_space

meta_file_path = "/home/dasb/workspace/QTT_SEG/src/finetune_wrapper/finetuning_results_sam.csv"
perf_predictor_path = "/home/dasb/.cache/qtt_new/PerfPredictor/"
cost_predictor_path = "/home/dasb/.cache/qtt_new/CostPredictor/"
output_path = "/work/dlclarge2/dasb-Camvid/config_checkpoints"
results_csv_path = "results.csv"

qtt_logs_path =  Path("/home/dasb/workspace/QTT_SEG/qtt/")
qtt_states_path = Path("/work/dlclarge2/dasb-Camvid/qtt_states/")

train_predictors = False
seed = 0
time_budgets = [180, 360, 540]
dataset_names =  ["leaf", "polyp", "eyes", "lesion", "fiber", "building", "cholec", "golf", "human_parsing", "terrain", "US"]

def delete_folders():
    print("Flushing stale files...")
    folder_paths = [
        qtt_logs_path,
        qtt_states_path,
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


if __name__ == "__main__":
    meta_df = pd.read_csv(meta_file_path)
    
    cost = meta_df["cost"]
    curve = meta_df.filter(regex=r'^epoch_\d{1,2}_iou$')
    config = meta_df.drop(columns=["cost", "dataset"] + curve.columns.tolist())
    X = config
    y = curve.values

    fit_params = {
        "batch_size": 4,
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

    for dataset_name in dataset_names:        
        delete_folders()
        cs = get_config_space()
        optimizer = QuickOptimizer(
            cs,
            max_fidelity=10,
            perf_predictor=perf_predictor,
            cost_predictor=cost_predictor,
            cost_aware=True,
            cost_factor=1.0,
            acq_fn="ei",
            patience=5,
            tol=0.001,
            refit=False,
            path=f"{qtt_states_path}{uuid.uuid4().hex}",
            seed=seed,
            verbosity = -1 
        )

        task_info = {
            "dataset_name": dataset_name,
            "output_path": output_path,
            "seed" : seed
        }

        metafeat = get_meta_data(dataset_name)

        print("Optimiser Setup")
        optimizer.setup(128, metafeat)

        tuner = QuickTuner(
            optimizer,
            finetune_script,
        )

        for time_budget in time_budgets:
            traj, runtime, history = tuner.run(fevals=100, trial_info=task_info, time_budget=time_budget)
            config_id, config, score, budget, cost, info = tuner.get_incumbent()
            df = pd.DataFrame([
                {
                    'dataset': entry['dataset'],
                    'config_id': entry['config-id'],
                    'score': entry['score'],
                    'cost' : entry['cost']
                }
                for entry in history
            ])
            df.to_csv(f"qtt_history_logs/{dataset_name}_{time_budget}.csv")

            result = {
                "DATASET": dataset_name,
                "SEED" : seed,
                "QTT_CONFIG": config,
                "QTT_TUNING_SCORE": score,
                "TIME_BUDGET": time_budget
            }
            if os.path.exists(results_csv_path):
                pd.DataFrame([result]).to_csv(results_csv_path, mode='a', index=False, header=False)
            else:
                pd.DataFrame([result]).to_csv(results_csv_path, index=False)