from src.sam2_process.sam2_train2 import main
from src.sam2_process.sam2_test import test
from src.utils.utils import get_parser
import pandas as pd
import os

# forest, flood, vineyard doesnt work on autogluon

time_budgets = [540]
best_scores_path = "SAM2_best_for_results.csv"

parser = get_parser() 
args = parser.parse_args()

for t in time_budgets:
    report = main(args = args, max_time = t)
    zero_shot_iou = test(zero_shot = True,args=args, save_images = True)

    report = {
        "Dataset" : report["dataset"] ,
        "Time Budget" : t,
        "Zero Shot IOU" : zero_shot_iou,
        "QTT IOU" : report["score"] 
    }
    if os.path.exists(best_scores_path):
        pd.DataFrame([report]).to_csv(best_scores_path, mode='a', index=False, header=False)
    else:
        pd.DataFrame([report]).to_csv(best_scores_path, index=False)

