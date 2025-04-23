import pandas as pd
import json
from autogluon.multimodal import MultiModalPredictor
import uuid
import os
import numpy as np

if __name__ == "__main__":
    time_budgets = [60, 120, 240, 300, 600]
    dataset_names = ["human_parsing_dataset", "semantic-drone-dataset", "FoodSeg103"]

    benchmark_file = "benchmarks/gluon.csv"
    if not os.path.exists(benchmark_file):
        pd.DataFrame(columns=["dataset_name", "time_budget", "score"]).to_csv(benchmark_file, index=False)

    for dataset_name in dataset_names:
        train_df = pd.read_csv(f"benchmarks/dataframes/{dataset_name}_train.csv")
        test_df = pd.read_csv(f"benchmarks/dataframes/{dataset_name}_test.csv")
        
        train_df = train_df.rename(columns={'image_paths': 'image', 'mask_paths': 'label'})
        test_df = test_df.rename(columns={'image_paths': 'image', 'mask_paths': 'label'})
            

        save_path = f"/work/dlclarge2/dasb-Camvid/autogluon_tmp/{uuid.uuid4().hex}-automm_semantic_seg"
        
        validation_metric = "iou"  
        checkpoint_name = "facebook/sam-vit-base"  
        efficient_finetune = "lora"  

        for time_budget in time_budgets:
            predictor = MultiModalPredictor(
                problem_type="semantic_segmentation",
                label="label",
                validation_metric=validation_metric,
                eval_metric=validation_metric,
                hyperparameters={
                    "env.precision": 32,
                    "model.sam.checkpoint_name": checkpoint_name,
                    "optimization.loss_function": "mask2former_loss",
                    "optimization.efficient_finetune": efficient_finetune,
                    "model.sam.num_mask_tokens": 10,
                },
                sample_data_path=train_df, 
                path=f"/work/dlclarge2/dasb-Camvid/{uuid.uuid4().hex}"
            )
            
            predictor.fit(train_data=train_df, tuning_data=test_df, time_limit=time_budget)
            
            np.random.seed(42)
            results = predictor.evaluate(test_df, metrics=[validation_metric])
            
            report = {
                "dataset_name" : dataset_name,
                "time_budget" : time_budget,
                "score" : results["iou"]
            }
            print(report)
            pd.DataFrame([report]).to_csv(benchmark_file, mode='a', index=False, header=False)



