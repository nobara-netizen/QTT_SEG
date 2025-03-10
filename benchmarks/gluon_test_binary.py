from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal import MultiModalPredictor
import uuid
import pandas as pd
import os
from PIL import Image
import numpy as np

if __name__ == "__main__":
    time_budgets = [60, 120, 240, 300, 600]
    dataset_names = ["human", "building", "lesion", "covid", "tree"]   
    benchmark_file = "benchmarks/gluon_iters.csv"
    if not os.path.exists(benchmark_file):
        pd.DataFrame(columns=["dataset_name", "time_budget", "score"]).to_csv(benchmark_file, index=False)

    for _ in range(10):
        for dataset_name in dataset_names:
            for time_budget in time_budgets:
                train_data = pd.read_csv(f"benchmarks/dataframes/{dataset_name}_train.csv")
                test_data = pd.read_csv(f"benchmarks/dataframes/{dataset_name}_test.csv")
                
                train_data = train_data.rename(columns={'image_paths': 'image', 'mask_paths': 'label'})
                test_data = test_data.rename(columns={'image_paths': 'image', 'mask_paths': 'label'})
                
                
                save_path = f"/work/dlclarge2/dasb-Camvid/autogluon/tmp/{uuid.uuid4().hex}-automm_semantic_seg"
                predictor = MultiModalPredictor(
                    problem_type="semantic_segmentation", 
                    label="label",
                    hyperparameters={
                            "model.sam.checkpoint_name": "facebook/sam-vit-base",
                        },
                    path=save_path,
                )
                predictor.fit(
                    train_data=train_data,
                    tuning_data=test_data,
                    time_limit=time_budget,
                )

                scores = predictor.evaluate(test_data, metrics=["iou"])
                print("Finetuned AG Score: ", scores)

                report = {
                    "dataset_name" : dataset_name,
                    "time_budget" : time_budget,
                    "score" : scores
                }
                pd.DataFrame([report]).to_csv(benchmark_file, mode='a', index=False, header=False)






