from pathlib import Path
import argparse
import json
import os
import time
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
from statistics import mean
from torch.utils.data import Dataset, DataLoader, Subset
import torch
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset
from src.sam2_process.sam2_test import test
from src.utils.utils import TARGET_MODULES_DICT, get_parser, set_seed
import random
from transformers import SamProcessor
import math
from torchmetrics.classification import JaccardIndex

sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"


def main(args,max_time=10**18):
    best_score = -float("inf")
    patience_counter = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jaccard = JaccardIndex(task="binary").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    train_dataset = CustomDataset(
        dataset_name = args.dataset_name, 
        processor = processor,
        train=True,
        resize_mask = False,
        args=args
        )    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device) 
    predictor = SAM2ImagePredictor(sam2_model)
    
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,  
            lora_alpha=2 * args.lora_rank,
            target_modules=["attn.qkv"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        peft_model = get_peft_model(sam2_model, lora_config)
        predictor.model = peft_model
        for name, param in predictor.model.named_parameters():
            if "mask_decoder" in name:
                param.requires_grad = True
    else:
        for param in predictor.model.parameters():
            param.requires_grad = False
        for name, param in predictor.model.named_parameters():
            if "mask_decoder" in name:
                param.requires_grad = True
                

    # Step 3: Confirm which parameters are trainable
    # print("Trainable parameters:")
    # for name, param in predictor.model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    # Step 4: Create optimizer using only trainable params
    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, predictor.model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_dataset) / 1)
    scheduler_cls = {
        "cosine": lambda opt: CosineAnnealingLR(opt, T_max=args.num_train_epochs),
        "plateau": lambda opt: ReduceLROnPlateau(
            opt, mode="max", patience=args.patience_epochs, factor=args.decay_rate
        ),
        "onecycle": lambda opt: OneCycleLR(
            opt, max_lr=args.lr, epochs=args.num_train_epochs, steps_per_epoch=steps_per_epoch
        )
    }.get(args.sched)

    scheduler = scheduler_cls(optimizer) if scheduler_cls else None
    scaler = torch.cuda.amp.GradScaler()
    output_dir = "output_masks_sam2"
    os.makedirs(output_dir, exist_ok=True)

    loss_fn = nn.BCEWithLogitsLoss()
    scores = {}
    start_time = time.time()

    for epoch in range(args.num_train_epochs):
        batch_iou = epoch_iou = avg_iou = 0
        batch_loss = epoch_loss = avg_loss = 0

        for _, batch in tqdm(enumerate(train_dataset), total=len(train_dataset), desc="Training Progress", leave=True):
            with torch.cuda.amp.autocast():
                if len(batch) == 0:
                    continue
                image, mask, input_point, input_label = batch["pixel_values"], batch["ground_truth_mask"], batch["input_points"], batch["input_labels"]

                predictor.set_image(image)

                # Prompt encoding
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    point_coords=input_point, point_labels=input_label, box=None, mask_logits=None, normalize_coords=True
                )
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )

                # Mask decoder
                batched_mode = unnorm_coords.shape[0] > 1  # Multi-object prediction
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
                prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

                gt_mask = torch.tensor(mask).unsqueeze(0).float().to(device)
                prd_mask = torch.sigmoid(prd_masks[:, 0])

                # loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
                
                loss = loss_fn(prd_masks[:, 0], gt_mask)
                batch_loss += loss.item()

                # inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                # iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
                gt_mask_bin = gt_mask.int()
                prd_mask_bin = (prd_mask > 0.5).int()

                iou = jaccard(prd_mask_bin, gt_mask_bin)

                batch_iou += iou.mean().item()

                # print(f"[DEBUG] Loss type: {type(loss)}, requires_grad: {loss.requires_grad}")

                predictor.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        epoch_iou = batch_iou / len(train_dataset)
        avg_iou += epoch_iou

        epoch_loss = batch_loss / len(train_dataset)
        avg_loss += epoch_loss

        score = test(zero_shot=False, predicted_model=predictor.model, args=args)
        print(f"Epoch: {epoch} IOU: {score}")
        
        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(predictor.model.state_dict(), os.path.join(args.output_dir, "sam2model.torch"))
        # else:
        #     patience_counter += 1
        #     if patience_counter >= args.patience_epochs:
        #         print(f"Early stopping at epoch {epoch} due to no improvement.")
        #         break
        scores[f"epoch_{epoch}_iou"] = best_score
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(score)
        elif scheduler:
            scheduler.step()
        
        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time} seconds reached. Stopping training early.")
            break  # Exit the loop when time limit is reached

    avg_iou /= args.num_train_epochs
    avg_loss /= args.num_train_epochs
    cost = time.time() - start_time

    report = {
        "dataset": args.dataset_name,
        "score": best_score,
        "cost": cost
    }
    if args.return_scores_per_epoch:
        return report, scores
    return report

if __name__ == "__main__":
    parser = get_parser() 
    args = parser.parse_args()
    report = main(args)
    

