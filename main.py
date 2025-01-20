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

from src.finetune_wrapper.finetune_script import finetune_script
from src.sam2_process.test import test


import pandas as pd
import numpy as np
import os

output_path = "/work/dlclarge2/dasb-Camvid/config_checkpoints"
results_csv_path = "results_final.csv"
all_columns = ["DATASET","QTT_CONFIG", "QTT_TUNING_SCORE", "TIME_BUDGET","QTT_TEST_DICE", "ZERO_SHOT_DICE","TEST IMPROVEMENT"]
time_budgets = [60, 120, 240, 300, 600]
dataset_names = ["human_parsing_dataset","sidewalk-semantic",  "danish-golf-courses-orthophotos", "semantic-drone-dataset", "FoodSeg103"]

def get_config_space():
    cs = ConfigurationSpace("cv-segmentation")

   
    # Learning rate and weight decay
    lr = OrdinalHyperparameter("learning_rate", [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01])
    wd = OrdinalHyperparameter("weight_decay", [0, 1e-05, 0.0001, 0.001, 0.01, 0.1])
    
    # Model selection
    model = Categorical("model_name", ["SAM"])

    # LoRA hyperparameters
    lora = OrdinalHyperparameter("lora", [0, 1])
    lora_targets = Categorical("lora_targets", ["image_enc_attn", "image_enc_mlp", "memory_self_attn", "memory_cross_attn"])
    lora_rank = OrdinalHyperparameter("lora_rank", [2, 4, 8])  # Assuming practical values
    lora_alpha = OrdinalHyperparameter("lora_alpha", [8, 16, 32])
    lora_dropout = OrdinalHyperparameter("lora_dropout", [0.0, 0.1, 0.2])

    # Augmentation hyperparameters
    horizontal_flip = OrdinalHyperparameter("horizontal_flip", [0, 1])
    vertical_flip = OrdinalHyperparameter("vertical_flip", [0, 1])
    random_rotate = OrdinalHyperparameter("random_rotate", [0, 1])
    elastic_transform = OrdinalHyperparameter("elastic_transform", [0, 1])
    normalize = OrdinalHyperparameter("normalize", [0, 1])

    # Optimizer arguments
    opt = Categorical("opt", ["sgd", "adam", "adamw"])
    momentum = OrdinalHyperparameter("momentum", [0.8, 0.9, 0.99])  # Only relevant for SGD
    opt_betas = Categorical("opt_betas", [(0.9, 0.999), (0.85, 0.995)])  # Example betas

    # Scheduler arguments
    sched = Categorical("sched", ["cosine", "step", "multistep", "plateau"])
    decay_epochs = OrdinalHyperparameter("decay_epochs", [10, 20, 30])
    decay_rate = OrdinalHyperparameter("decay_rate", [0.001, 0.01, 0.1])



    # Adding hyperparameters to configuration space
    cs.add_hyperparameters([
        lr, wd, model, lora, lora_targets, lora_rank, lora_alpha, lora_dropout,
        horizontal_flip, vertical_flip, random_rotate, elastic_transform, normalize,
        opt, momentum, opt_betas, sched, decay_epochs, decay_rate,
    ])

    return cs

if __name__ == "__main__":

    if not os.path.exists(results_csv_path):
        pd.DataFrame(columns=[all_columns]).to_csv(results_csv_path, index=False)

    train_predictors = False

    perf_predictor_path = "/home/dasb/.cache/qtt_new/PerfPredictor/"
    cost_predictor_path = "/home/dasb/.cache/qtt_new/CostPredictor/"

    df = pd.read_csv("src/finetune_wrapper/finetuning_results.csv")
    cost = df["cost"]
    curve = df.filter(regex=r'^epoch_\d{1,2}$')
    config = df.drop(columns=["cost" ,"score"] + curve.columns.tolist())

    X = config
    y = curve.values

    fit_params = {
        "batch_size": 100,
    }
    

    for _ in range(5):
        for time_budget in time_budgets:
            for dataset_name in dataset_names:

                if train_predictors:

                    perf_predictor = PerfPredictor(fit_params).fit(X,y)
                    perf_predictor.save(perf_predictor_path)
                
                    y = cost.values.reshape(-1, 1) 

                    cost_predictor = CostPredictor(fit_params).fit(X,y)
                    cost_predictor.save(cost_predictor_path)

                else:    
                    perf_predictor = PerfPredictor().load(perf_predictor_path)
                    cost_predictor = CostPredictor().load(cost_predictor_path)

                print("Generate Optimiser")
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
                    path = "/work/dlclarge2/dasb-Camvid/qtt_opt_state"                           
                )

                task_info = {
                "dataset_name" : dataset_name,
                "output_path" : output_path,
                "root" : "/work/dlclarge2/dasb-Camvid/qtt_seg_datasets"
                }

                print("Optimiser Setup")
                optimizer.setup(100)  

                tuner = QuickTuner(
                    optimizer,
                    finetune_script,  
                )

                traj, runtime, history = tuner.run(fevals=100,trial_info=task_info,  time_budget=time_budget)
                config_id, config, score, budget, cost, info = tuner.get_incumbent()


                print()
                print("*****   PERFORMANCE COMPARISON QTT-SEG   *****")
                print("=================================")  

                zero_shot_dice = test(
                    dataset_name, 
                    root = task_info["root"],
                    max_iters = 10,
                    zero_shot = True
                    )

                qtt_dice = test(
                    dataset_name,
                    root = task_info["root"],
                    max_iters=10,
                    zero_shot=False,
                    predicted_model_path=f"{info['path']}/sam2model.torch",  
                    args=config
                )
                
                dice_imp = ((qtt_dice - zero_shot_dice) / zero_shot_dice) * 100

                print("Dice Improvement: {:.2f}%".format(dice_imp))
                print("---------------------------------")
                
                result = {
                    "DATASET" : dataset_name,
                    "QTT_CONFIG": config, 
                    "QTT_TUNING_SCORE": score*100, 
                    "TIME_BUDGET" : time_budget,
                    "QTT_TEST_DICE" : qtt_dice,
                    "ZERO_SHOT_DICE" : zero_shot_dice,
                    "TEST IMPROVEMENT" : dice_imp
                }
                pd.DataFrame([result]).to_csv(results_csv_path, mode='a', index=False, header=False)