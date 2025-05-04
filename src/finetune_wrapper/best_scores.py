from src.sam2_process.sam2_train import main
from src.sam2_process.sam2_test import test
from src.utils.utils import get_parser
import pandas as pd
import os
import argparse
import pandas as pd

time_budget = 180
best_scores_path = "SAM2_best_for_results.csv"

def get_best_pipeline_args(df):
    # Group by dataset (if more than one), get row with highest epoch_9_iou
    best_rows = df.loc[df.groupby("dataset")["epoch_9_iou"].idxmax()]
    
    best_args = []
    for _, row in best_rows.iterrows():
        args = {
            "--dataset_name": row["dataset"],
            "--lr": float(row["lr"]),
            "--weight_decay": float(row["weight_decay"]),
            "--lora": int(row["lora"]),
            "--lora_rank": int(row["lora_rank"]),
            "--lora_dropout": float(row["lora_dropout"]),
            "--sched": row["sched"],
            "--decay_rate": float(row["decay_rate"]),
            "--patience_epochs": int(row["patience_epochs"]),
            "--cosine_t0": int(row["cosine_t0"]),
            "--cosine_t_mult": int(row["cosine_t_mult"]),
            "--onecycle_pct_start": float(row["onecycle_pct_start"]),
            "--onecycle_div_factor": float(row["onecycle_div_factor"]),
            "--onecycle_final_div_factor": float(row["onecycle_final_div_factor"]),
            "--step_size": int(row["step_size"]),
            "--poly_power": float(row["poly_power"]),
            "--horizontal_flip": int(row["horizontal_flip"]),
            "--vertical_flip": int(row["vertical_flip"]),
            "--random_rotate": int(row["random_rotate"]),
        }
        best_args.append(args)
    return best_args

if __name__ == "__main__":
    df = pd.read_csv("your_dataset.csv")  
    best_args_list = get_best_pipeline_args(df)
    parser = get_parser()
    for best_args in best_args_list:
        print("\nBest config for dataset:", best_args["--dataset_name"])
        args = parser.parse_args(sum([[k, str(v)] for k, v in best_args.items()], []))
        report = main(args = args, max_time = time_budget)
        zero_shot_iou = test(zero_shot = True,args=args, save_images = True)
        report = {
            "Dataset" : report["dataset"] ,
            "Time Budget" : time_budget,
            "Zero Shot IOU" : zero_shot_iou,
            "QTT IOU" : report["score"] 
        }
        if os.path.exists(best_scores_path):
            pd.DataFrame([report]).to_csv(best_scores_path, mode='a', index=False, header=False)
        else:
            pd.DataFrame([report]).to_csv(best_scores_path, index=False)





# parser = get_parser() 
# args = parser.parse_args()

# report = main(args = args, max_time = time_budget)
# zero_shot_iou = test(zero_shot = True,args=args, save_images = True)

# report = {
#     "Dataset" : report["dataset"] ,
#     "Time Budget" : time_budget,
#     "Zero Shot IOU" : zero_shot_iou,
#     "QTT IOU" : report["score"] 
# }
# if os.path.exists(best_scores_path):
#     pd.DataFrame([report]).to_csv(best_scores_path, mode='a', index=False, header=False)
# else:
#     pd.DataFrame([report]).to_csv(best_scores_path, index=False)

