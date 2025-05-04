import numpy as np
import torch
import cv2
from statistics import mean
import os
import json
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset
from transformers import SamProcessor
from torch.utils.data import Subset
import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torchmetrics.classification import JaccardIndex
from src.utils.utils import get_parser

sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

def test(
        zero_shot=False,
        split = "test",
        predicted_model=None, 
        predicted_model_path = None,
        args=None,
        save_images = False
):
    device = "cuda"
    jaccard = JaccardIndex(task="binary")
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device)
    test_dataset = CustomDataset(
        dataset_name = args.dataset_name, 
        split=split,
        args=args
        )
    predictor = SAM2ImagePredictor(sam2_model)

    if args:
        if args.lora:
            lora_config = LoraConfig(
                r=args.lora_rank,  
                lora_alpha=2 * args.lora_rank,
                target_modules=["attn.qkv","mlp.layers.0", "mlp.layers.1"],
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION 
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

    if save_images:
        if zero_shot:
            output_dir = f"Best_Scores/{args.dataset_name}_zero_shot"
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = f"Best_Scores/{args.dataset_name}_qtt"
            os.makedirs(output_dir, exist_ok=True)

    images_saved = 0
    with torch.no_grad(): 
        for idx, batch in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Testing Progress", leave=True):
            if len(batch) == 0:
                print(f"idx {idx} skipped")
                continue
            
            image, mask, input_box = batch["pixel_values"], batch["ground_truth_mask"], batch["input_box"]
            
            predictor.set_image(image)
            prd_masks, _, _ = predictor.predict(
                box = input_box
            )   
            
            gt_mask = torch.tensor(mask).float()
            if len(gt_mask.shape)==2:
                prd_mask = prd_masks[0]
            else:
                prd_mask = prd_masks[:,0]
            prd_mask = torch.sigmoid(torch.from_numpy(prd_mask))
            # print(image.shape, mask.shape, input_box.shape,gt_mask.shape, prd_mask.shape)

            gt_mask_bin = gt_mask.int()
            prd_mask_bin = (prd_mask > 0.5).int()

            iou = jaccard(prd_mask_bin, gt_mask_bin)

            mean_iou += iou.mean().item()

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
    parser = get_parser() 
    args = parser.parse_args()
    mean_score = test(
        zero_shot = True,
        args = args
    )
    print(mean_score)