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
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, 
    OneCycleLR,
    StepLR,
    PolynomialLR
)
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset
from src.sam2_process.sam2_test import test
from src.utils.utils import get_parser, plot_training_metrics
import random
from transformers import SamProcessor
import math
from torchmetrics.classification import JaccardIndex
import contextlib
import os
import sys

sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        smooth = 1e-5
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return bce_loss + dice_loss

def numpy_collate(batch):
    return batch  

def main(args, max_time=10**18):
    best_score = -float("inf")
    test_score = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jaccard = JaccardIndex(task="binary").to(device)
    output_dir = os.path.join(args.output_dir, "sam2model.torch")
    
    train_dataset = CustomDataset(dataset_name=args.dataset_name, split="train", args=args)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=numpy_collate)
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device) 
    predictor = SAM2ImagePredictor(sam2_model)

    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules=["attn.qkv", "mlp.layers.0", "mlp.layers.1"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        peft_model = get_peft_model(sam2_model, lora_config)
        predictor.model = peft_model
        for name, param in predictor.model.named_parameters():
            if "mask_decoder" in name or "prompt_encoder" in name:
                param.requires_grad = True
    else:
        for param in predictor.model.parameters():
            param.requires_grad = False
        for name, param in predictor.model.named_parameters():
            if "mask_decoder" in name or "prompt_encoder" in name:
                param.requires_grad = True

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, predictor.model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    steps_per_epoch = len(train_loader)
    scheduler_cls = {
        "cosine_warm": lambda opt: CosineAnnealingWarmRestarts(
            opt, T_0=args.cosine_t0, T_mult=args.cosine_t_mult, eta_min=args.lr / 10
        ),
        "cosine": lambda opt: CosineAnnealingLR(
            opt, T_max=args.num_train_epochs, eta_min=args.lr / 10
        ),
        "plateau": lambda opt: ReduceLROnPlateau(
            opt, mode="max", patience=args.patience_epochs, factor=args.decay_rate
        ),
        "onecycle": lambda opt: OneCycleLR(
            opt,
            max_lr=args.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=args.num_train_epochs,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
            anneal_strategy='cos'
        ),
        "step": lambda opt: StepLR(
            opt, step_size=args.step_size, gamma=args.decay_rate, last_epoch=-1
        ),
        "poly": lambda opt: PolynomialLR(
            opt, total_iters=args.num_train_epochs * steps_per_epoch, power=args.poly_power
        )
    }.get(args.sched)

    scheduler = scheduler_cls(optimizer) if scheduler_cls else None

    loss_fn = BCEDiceLoss()
    lc = {}
    train_loss = []
    train_iou = []
    val_iou = []

    predictor.model.train()
    start_time = time.time()

    for epoch in range(args.num_train_epochs):
        batch_iou = batch_loss = 0

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Progress", leave=True):
            sample = batch[0]
            if len(batch) == 0:
                continue

            image, mask, input_box = sample["pixel_values"], sample["ground_truth_mask"], sample["input_box"]
            predictor.set_image(image)

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                point_coords=None, point_labels=None, box=input_box, mask_logits=None, normalize_coords=True
            )
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                points=None, boxes=unnorm_box, masks=None
            )

            batched_mode = unnorm_box.shape[0] > 1
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

            loss = loss_fn(prd_masks[:, 0], gt_mask)
            batch_loss += loss.item()

            gt_mask_bin = gt_mask.int()
            prd_mask_bin = (prd_masks[:, 0] > 0.5).int()

            iou = jaccard(prd_mask_bin, gt_mask_bin)
            batch_iou += iou.mean().item()

            predictor.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
            optimizer.step()

            if isinstance(scheduler, (OneCycleLR, PolynomialLR)):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step(epoch + i / steps_per_epoch)

        epoch_iou = batch_iou / len(train_dataset)
        train_iou.append(epoch_iou)

        epoch_loss = batch_loss / len(train_dataset)
        train_loss.append(epoch_loss)

        val_score = test(split="val", predicted_model=predictor.model, args=args)
        print(f"Epoch: {epoch} VAL IOU: {val_score} LR: {optimizer.param_groups[0]['lr']}")
        val_iou.append(val_score)

        if val_score > best_score:
            best_score = val_score
            # torch.save(predictor.model.state_dict(), output_dir)

        lc[f"epoch_{epoch}_iou"] = val_score

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_score)
        elif isinstance(scheduler, (StepLR, CosineAnnealingLR)):
            scheduler.step()

        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time} seconds reached. Stopping training early.")
            break

    plot_training_metrics(train_loss, train_iou, val_iou, save_path='training_metrics.png')

    avg_iou = sum(train_iou) / len(train_iou)
    avg_loss = sum(train_loss) / len(train_loss)
    cost = time.time() - start_time

    report = {
        "dataset": args.dataset_name,
        "score": test_score if test_score else best_score,
        "cost": cost
    }

    if args.return_scores_per_epoch:
        return report, lc
    return report

if __name__ == "__main__":
    parser = get_parser() 
    args = parser.parse_args()
    report = main(args)
    

