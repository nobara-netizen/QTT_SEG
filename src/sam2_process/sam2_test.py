import numpy as np
import torch
import cv2
from statistics import mean
import os
import json
from peft import get_peft_model, LoraConfig

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset

sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"


valid_lora_targets = {
    "image_enc_attn" : ["attn.qkv" , "attn.proj'"],
    "image_enc_mlp" : ["mlp.layers.0", "mlp.layers.1"],
    "memory_self_attn" : [ "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.out_proj"], 
    "memory_cross_attn" : ["cross_attn_image.q_proj","cross_attn_image.k_proj","cross_attn_image.v_proj","cross_attn_image.out_proj"] 
}

def dice_score(pred_mask, gt_mask):
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()

    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    dice = (2.0 * intersection) / union
    return dice

def dice_multiclass(pred, true):
        avg_score = 0
        for i in range(pred.shape[0]):
                per_mask_score = dice_score(pred[i], true[i])
                avg_score += per_mask_score
        return avg_score/pred.shape[0]

def test(
        dataset_name = "",
        zero_shot = True, 
        predictor = None, 
        predicted_model_path = None, 
        args=None,
):
        test_dataset = CustomDataset(dataset_name, train=False)

        if predictor is None:
                sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

                if zero_shot==True:
                        predictor = SAM2ImagePredictor(sam2_model)
                
                else:
                        if args["lora"]:
                                print("Applying LoRA")
                                target_modules = []
                                selected_modules = args["lora_targets"].split(",")
                                for module in selected_modules:
                                        if module in valid_lora_targets:  
                                                target_modules.extend(valid_lora_targets[module])  
                                        else:
                                                raise ValueError(f"Invalid LoRA target module: {module}")

                                lora_config = LoraConfig(
                                target_modules=target_modules,  
                                r=args["lora_rank"],  
                                lora_alpha=2 * args["lora_rank"],  
                                lora_dropout=args["lora_dropout"],  
                                )
                                sam2_model = get_peft_model(sam2_model, lora_config)  
                        
                        predictor = SAM2ImagePredictor(sam2_model)
                        predictor.model.load_state_dict(torch.load(predicted_model_path))
        
        predictor.model.eval()
        mean_iou = 0
        test_itr = 0

        with torch.no_grad(): 
                while test_itr < len(test_dataset):
                        image, mask, input_point, input_label = test_dataset[test_itr]
                        
                        if mask.shape == (0,):
                                continue

                        predictor.set_image(image)
                        prd_masks, _, _ = predictor.predict(
                                point_coords=input_point,
                                point_labels=input_label
                                )
                        if len(prd_masks.shape) == 4:
                                prd_mask = prd_masks[:, 0, :, :]
                        elif len(prd_masks.shape) == 3:
                                prd_mask = np.expand_dims(prd_masks[0], axis=0)
                        else:
                                raise ValueError(f"Expected 3 or more predicted masks, got {prd_masks.shape[0]}")
                        gt_mask = mask.astype(np.float32)
                        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                        iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                        
                        mean_iou += mean(iou)
                        test_itr += 1
                mean_iou /= len(test_dataset)
        return mean_iou

                
if __name__ == "__main__":

        mean_score = test(
                dataset_name="building",
                zero_shot = True,
        )
        print(mean_score)