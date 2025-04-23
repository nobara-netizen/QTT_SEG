import numpy as np
import torch
import cv2
from statistics import mean
import os
import json
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset
from src.utils.utils import TARGET_MODULES_DICT
from transformers import SamProcessor
from torch.utils.data import Subset
import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchmetrics.classification import JaccardIndex

sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

def test(
        zero_shot,
        predicted_model_path = None,
        predicted_model=None, 
        args=None,
        save_images = False
):
    device = "cuda"
    jaccard = JaccardIndex(task="binary")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = CustomDataset(args.dataset_name, processor=processor, resize_mask=False, train=False, args=args)
    predictor = SAM2ImagePredictor(sam2_model)

    if args:
        if args.lora:
            lora_config = LoraConfig(
                target_modules=TARGET_MODULES_DICT[args.lora_targets],  
                r=args.lora_rank,  
                lora_alpha=2 * args.lora_rank,  
                lora_dropout=args.lora_dropout,  
            )
            peft_model = get_peft_model(sam2_model, lora_config)
            predictor.model = peft_model 

    
    if predicted_model_path:
        predictor.model.load_state_dict(torch.load(predicted_model_path, weights_only=True))
        print("Loaded Model State")
    
    if predicted_model:
        predictor.model = predicted_model
        print("Loaded Model")

    predictor.model.eval()
    mean_iou = 0

    # Create output directory
    if save_images:
        if zero_shot:
            output_dir = f"Best_Scores/{args.dataset_name}_zero_shot"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = f"Best_Scores/{args.dataset_name}_qtt"
            os.makedirs(output_dir, exist_ok=True)

    images_saved = 0
    with torch.no_grad(): 
        # for idx, batch in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Testing Progress", leave=True):
        for idx, batch in enumerate(test_dataset):
            if len(batch) == 0:
                print(f"idx {idx} skipped")
                continue
            
            image, mask, input_point, input_label = batch["pixel_values"], batch["ground_truth_mask"], batch["input_points"], batch["input_labels"]

            predictor.set_image(image)
            prd_masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label
            )   
            
            gt_mask = torch.tensor(mask).float()
            prd_mask = torch.sigmoid(torch.from_numpy(prd_masks[0]))

            gt_mask_bin = gt_mask.int()
            prd_mask_bin = (prd_mask > 0.5).int()

            iou = jaccard(prd_mask_bin, gt_mask_bin)

            # inter = (gt_mask * (prd_mask > 0.5)).sum(0).sum(0)
            # iou = inter / (gt_mask.sum(0).sum(0) + (prd_mask > 0.5).sum(0).sum(0) - inter)
            mean_iou += iou.mean().item()

            # Save only the first 5 images
            if save_images:
                if images_saved < 5:
                    image_pil = transforms.ToPILImage()(image)
                    gt_mask_pil = transforms.ToPILImage()(gt_mask)
                    prd_mask_pil = transforms.ToPILImage()((prd_mask > 0.5).float())

                    image_pil.save(os.path.join(output_dir, f"image_{images_saved}.png"))
                    gt_mask_pil.save(os.path.join(output_dir, f"gt_mask_{images_saved}.png"))
                    prd_mask_pil.save(os.path.join(output_dir, f"prd_mask_{images_saved}.png"))

                    images_saved += 1

    mean_iou /= len(test_dataset)
    return mean_iou

                
if __name__ == "__main__":

        mean_score = test(
                dataset_name="building",
                zero_shot = True,
        )
        print(mean_score)