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
from peft import get_peft_model, LoraConfig
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset
from src.sam2_process.sam2_test import test
from src.utils.utils import set_seed, plot_graph

sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"


valid_lora_targets = {
    "image_enc_attn" : ["attn.qkv" , "attn.proj'"],
    "image_enc_mlp" : ["mlp.layers.0", "mlp.layers.1"],
    "memory_self_attn" : [ "self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.out_proj"], 
    "memory_cross_attn" : ["cross_attn_image.q_proj","cross_attn_image.k_proj","cross_attn_image.v_proj","cross_attn_image.out_proj"] 
}


def get_parser():
    parser = argparse.ArgumentParser()

    # Fixed args
    parser.add_argument("--dataset_name", default="building", type=str, help="Dataset Name")
    parser.add_argument("--seed", default=0, type=int, help="Seed for sampling")
    parser.add_argument("--output_dir", default="./outputs", type=str, help="Output path")
    parser.add_argument("--num_train_epochs", default=50, type=int, help="Number of training epochs")
    parser.add_argument('--return_scores_per_epoch', action='store_true', help="Return scores per epoch")

    # Hyperparameters
    parser.add_argument("--lr", default=0.00001, type=float, 
                        choices=[0.01, 0.001, 0.0001, 0.00001, 0.000001], 
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, 
                        choices=[1e-4, 1e-5], 
                        help="Weight decay (L2 regularization)")

    # Optimizer arguments
    parser.add_argument("--opt", choices=["sgd", "adam", "adamw", "rmsprop"], 
                        default="adam", help="Optimizer type")
    parser.add_argument("--momentum", default=0.9, type=float, 
                        choices=[0.8, 0.9], 
                        help="Momentum (for SGD only)")
    parser.add_argument("--opt_betas", type=eval, default=(0.9, 0.999), 
                        choices=[(0.9, 0.999), (0.85, 0.995), (0.8, 0.9)], 
                        help="Betas for Adam/AdamW (Tuple)")

    # Scheduler arguments
    parser.add_argument("--sched", choices=["cosine", "plateau","onecycle"], 
                        default=None, help="Learning rate scheduler")
    parser.add_argument("--decay_epochs", default=30, type=int, 
                        choices=[30, 60, 90], 
                        help="Epochs before applying decay")
    parser.add_argument("--decay_rate", default=0.001, type=float, 
                        choices=[0.1, 0.3, 0.5], 
                        help="Decay rate")
    parser.add_argument("--patience_epochs", default=5, type=int, help="Patience epochs for plateau scheduler")

    # AMP setup
    parser.add_argument("--amp", action='store_true', help="Enable Automatic Mixed Precision")

    # LoRA Hyperparameters
    parser.add_argument("--lora", default=0, type=int, choices=[0, 1], help="Enable LoRA")
    parser.add_argument("--lora_targets", type=str, 
                        choices=["image_enc_attn", "image_enc_mlp", "memory_self_attn", "memory_cross_attn"], 
                        default="image_enc_attn", help="LoRA targets")
    parser.add_argument("--lora_rank", default=4, type=int, 
                        choices=(4, 8), 
                        help="LoRA rank")
    
    parser.add_argument("--lora_dropout", default=0.1, type=float, 
                        choices=(0.1,0.2, 0.3), 
                        help="LoRA dropout")

    # Augmentation Hyperparameters
    parser.add_argument("--horizontal_flip", default=0, type=int, choices=[0, 1], help="Enable horizontal flip augmentation")
    parser.add_argument("--vertical_flip", default=0, type=int, choices=[0, 1], help="Enable vertical flip augmentation")
    parser.add_argument("--random_rotate", default=0, type=int, choices=[0, 1], help="Enable random rotation augmentation")
    parser.add_argument("--elastic_transform", default=0, type=int, choices=[0, 1], help="Enable elastic transform augmentation")
    parser.add_argument("--normalize", default=0, type=int, choices=[0, 1], help="Enable normalization of images")

    return parser



def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)  
    whole_dataset = CustomDataset(args.dataset_name, train=True, args=args)

    sampled_indices = torch.randperm(len(whole_dataset)).tolist()[:100]  
    dataset = Subset(whole_dataset, sampled_indices)  

    # Apply LoRA if enabled
    if args.lora:
        print("Applying LoRA")
        target_modules = []
        selected_modules = args.lora_targets.split(",")
        for module in selected_modules:
            if module in valid_lora_targets:  
                target_modules.extend(valid_lora_targets[module])  
            else:
                raise ValueError(f"Invalid LoRA target module: {module}")

        lora_config = LoraConfig(
            target_modules=target_modules,  
            r=args.lora_rank,  
            lora_alpha=2 * args.lora_rank,  
            lora_dropout=args.lora_dropout,  
        )
        sam2_model = get_peft_model(sam2_model, lora_config)
        sam2_model.print_trainable_parameters()

    predictor = SAM2ImagePredictor(sam2_model)

    for name, param in predictor.model.named_parameters():
        if "sam_mask_decoder" in name or "sam_prompt_encoder" in name:
            param.requires_grad = True

    # Define optimizer
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
        params=predictor.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **({"momentum": args.momentum} if args.opt == "sgd" else {"betas": opt_betas} if "adam" in args.opt else {})
    )

    # AMP setup
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Scheduler setup
    scheduler_cls = {
        "cosine": lambda opt: CosineAnnealingLR(opt, T_max=args.num_train_epochs),
        "plateau": lambda opt: ReduceLROnPlateau(opt, patience=args.patience_epochs, factor=args.decay_rate),
        "onecycle": lambda opt: OneCycleLR(opt, max_lr=args.lr, epochs=args.num_train_epochs, steps_per_epoch=len(dataset))
    }.get(args.sched, None)

    scheduler = scheduler_cls(optimizer) if scheduler_cls else None

    # Training loop
    start_time = time.time()
    losses, ious = [], {}
    
    for epoch in range(args.num_train_epochs):
        epoch_loss, epoch_iou = 0, 0
        num_samples = 0
        
        for i in tqdm(range(len(dataset))):
            torch.cuda.empty_cache()
            
            image, mask, input_point, input_label = dataset[i]
            if mask.shape == (0,):
                continue  
            
            predictor.set_image(image)

            with torch.no_grad():
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
                )
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )
                high_res_features = [
                    feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
                ]
            
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=(unnorm_coords.shape[0] > 1),
                high_res_features=high_res_features,
            )

            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
            
            gt_mask = torch.tensor(mask.astype(np.float32), device=device)
            prd_mask = torch.sigmoid(prd_masks[:, 0])

            # Compute loss
            loss = nn.BCEWithLogitsLoss()(prd_mask, gt_mask.float())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Scheduler step
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(loss)
            elif scheduler:
                scheduler.step()

            # IoU Calculation
            inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1, 2))
            union = gt_mask.sum(dim=(1, 2)) + (prd_mask > 0.5).sum(dim=(1, 2)) - inter
            iou = inter / (union + 1e-6)  # Avoid division by zero
            
            epoch_loss += loss.item()
            epoch_iou += iou.mean().item()
            num_samples += 1
            
            # Move tensors to CPU & free memory
            prd_masks = prd_masks.detach().cpu()
            gt_mask = gt_mask.detach().cpu()
            torch.cuda.empty_cache()

        # Compute mean loss and IoU for epoch
        epoch_loss /= max(1, num_samples)
        epoch_iou /= max(1, num_samples)
        
        losses.append(epoch_loss)
        ious[f"epoch_{epoch}_iou"] = epoch_iou
        print(f"Epoch {epoch}: Loss: {epoch_loss:.4f}, IoU: {epoch_iou:.4f}")

    # Training cost calculation
    cost = time.time() - start_time
    report = {
        "dataset": args.dataset_name,
        "score": np.mean(list(ious.values())),
        "cost": cost
    }

    if args.return_scores_per_epoch:
        return report, ious
    return report

if __name__ == "__main__":
    parser = get_parser() 
    args = parser.parse_args()
    report = main(args)
    # print("Epochs: {}, Score: {}, Cost: {}".format(args.num_train_epochs, report["score"], report["cost"]))
    # print(test(args.dataset_name, zero_shot=True, args=args.__dict__))
    # print(test(args.dataset_name, zero_shot=False, predicted_model_path= os.path.join(args.output_dir, "sam2model.torch"), args=args.__dict__))


