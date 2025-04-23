import numpy as np 
from torch.utils.data import Dataset, DataLoader
from transformers import SamProcessor
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from src.data.custom_dataloader import CustomDataset
from torch.utils.data import Subset
import argparse
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
from transformers import SamModel 
import torch.nn as nn
from src.utils.utils import get_parser, set_seed
import time
from peft import LoraConfig, get_peft_model
import random
import os
from PIL import Image
import torchvision.transforms as transforms

def train(args):
    set_seed(args.seed)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_dataset = CustomDataset(
        args.dataset_name,
        processor, 
        train=True, 
        resize_mask=True,
        multiclass=False,
        args=args
    )
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    print(args.lora_targets)

    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,  
            target_modules=args.lora_targets, 
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)
    else:
        for name, param in model.named_parameters():
            if "mask_decoder" in name:
                param.requires_grad = True
            elif "prompt_encoder" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad == True:
            print(name)

    opt_betas = eval(args.opt_betas) if isinstance(args.opt_betas, str) else args.opt_betas
    optimizer_cls = {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "rmsprop": RMSprop
    }.get(args.opt, None)

    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer: {args.opt}")

    optimizer = optimizer_cls(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **({"momentum": args.momentum} if args.opt == "sgd" else {"betas": opt_betas} if "adam" in args.opt else {})
    )
    scheduler_cls = {
        "cosine": lambda opt: CosineAnnealingLR(opt, T_max=args.num_train_epochs),
        "plateau": lambda opt: ReduceLROnPlateau(opt, patience=args.patience_epochs, factor=args.decay_rate),
        "onecycle": lambda opt: OneCycleLR(opt, max_lr=args.lr, epochs=args.num_train_epochs, steps_per_epoch=len(dataset))
    }.get(args.sched, None)
    
    scheduler = scheduler_cls(optimizer) if scheduler_cls else None
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    scores = {}
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    output_dir = "output_masks_sam1"
    os.makedirs(output_dir, exist_ok=True)
    model.train()

    for epoch in range(args.num_train_epochs):
        # if time.time() - start_time > max_time:
        #     print(f"Time limit of {max_time} seconds reached. Stopping training early.")
        #     break

        batch_iou = epoch_iou = avg_iou = 0
        batch_loss = epoch_loss = avg_loss = 0

        for batch in tqdm(train_dataloader):
            if len(batch) == 0:
                continue
            
            with torch.amp.autocast("cuda"):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_points=batch["input_points"].to(device),
                    input_labels=batch["input_labels"].to(device),
                    multimask_output=False
                )
                # print(outputs.pred_masks.shape, batch["ground_truth_mask"].shape)
                prd_mask = torch.sigmoid(outputs.pred_masks.squeeze(2).to(device))
                gt_mask = batch["ground_truth_mask"].unsqueeze(1).to(device).float()

                # loss = loss_fn(prd_mask, gt_mask)
                loss = (-gt_mask * torch.log(prd_mask + 1e-5) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)).mean()
                
                batch_loss += loss.item()

                inter = (gt_mask * (prd_mask > 0.5)).sum(2).sum(2)
                iou = inter / (gt_mask.sum(2).sum(2) + (prd_mask > 0.5).sum(2).sum(2) - inter)
                batch_iou += iou.mean().item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch_iou = batch_iou / len(train_dataloader)
        avg_iou += epoch_iou
        scores[f"epoch_{epoch}_iou"] = epoch_iou

        epoch_loss = batch_loss / len(train_dataloader)
        avg_loss += epoch_loss

        # print(f"Epoch: {epoch} IOU: {epoch_iou}")
        test_iou = test(predicted_model=model, args=args)
        print(f"Epoch: {epoch} IOU: {test_iou}")
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        elif scheduler:
            scheduler.step()

    avg_iou /= args.num_train_epochs
    avg_loss /= args.num_train_epochs
    cost = time.time() - start_time
    
    report = {
        "dataset": args.dataset_name,
        "score": avg_iou,
        "cost": cost
    }
    
    if args.return_scores_per_epoch:
        return report, scores
    return report, model

def test(
        predicted_model, 
        args=None,
        save_images=True

):
    device = "cuda"
    if save_images:
        output_dir = f"SAM1_Scores/{args.dataset_name}_qtt"
        os.makedirs(output_dir, exist_ok=True)
    scaler = torch.amp.GradScaler("cuda")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    test_dataset = CustomDataset(
        args.dataset_name,
        processor, 
        train=False, 
        resize_mask=True,
        multiclass=False,
        args=args,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = predicted_model
    model.eval()
    model.to(device)
    mean_iou = 0
    images_saved = 0
    for idx, batch in enumerate(test_dataloader):
        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_points=batch["input_points"].to(device),
                    input_labels=batch["input_labels"].to(device),
                    multimask_output=False
                )

                prd_mask = torch.sigmoid(outputs.pred_masks.squeeze(2))
                gt_mask = batch["ground_truth_mask"].float().to(device).unsqueeze(1)
                inter = (gt_mask * (prd_mask > 0.5)).sum(2).sum(2)
                iou = inter / (gt_mask.sum(2).sum(2) + (prd_mask > 0.5).sum(2).sum(2) - inter)
                mean_iou += iou.mean().item()

                if save_images:
                    if images_saved < 5:
                        image_pil = transforms.ToPILImage()(batch["pixel_values"].squeeze(0))
                        gt_mask_pil = transforms.ToPILImage()(gt_mask.squeeze(0))
                        prd_mask_pil = transforms.ToPILImage()((prd_mask.squeeze(0) > 0.5).float())

                        image_pil.save(os.path.join(output_dir, f"image_{images_saved}.png"))
                        gt_mask_pil.save(os.path.join(output_dir, f"gt_mask_{images_saved}.png"))
                        prd_mask_pil.save(os.path.join(output_dir, f"prd_mask_{images_saved}.png"))

                        images_saved += 1
    mean_iou /= len(test_dataloader)

    return mean_iou



if __name__ == "__main__":
    parser = get_parser() 
    args = parser.parse_args()
    report = train(args=args)