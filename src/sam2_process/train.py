from pathlib import Path
import argparse
import json
import os
import time

import cv2
from PIL import Image
import numpy as np
from statistics import mean

import torch
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ReduceLROnPlateau

from peft import get_peft_model, LoraConfig
import matplotlib.pyplot as plt

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

def get_parser():
    parser = argparse.ArgumentParser()

    # Fixed args
    parser.add_argument("--dataset_name", default="FoodSeg103", type=str, help="Dataset Name")
    parser.add_argument("--output_dir", default="./outputs", type=str, help="output path")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="number of training epochs")
    parser.add_argument('--return_scores_per_epoch', action='store_true', help="Return scores per epoch")

    # Hyperparameters
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay (default: 1e-4)")
    
    # Optimizer arguments
    parser.add_argument("--opt", choices=["sgd", "adam", "adamw"], default="adam", help="Optimizer type")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum (for SGD only)")
    parser.add_argument("--opt_betas", default="(0.9, 0.999)", type=str, help="Betas for Adam/AdamW")
    
    # Scheduler arguments
    parser.add_argument("--sched", choices=["cosine", "step", "multistep", "plateau"], default=None, help="Learning rate scheduler")
    parser.add_argument("--decay_epochs", default=30, type=int, help="Epochs before applying decay")
    parser.add_argument("--decay_rate", default=0.001, type=float, help="Decay rate")
    parser.add_argument("--patience_epochs", default=5, type=int, help="Patience epochs for plateau scheduler")
    
    # AMP setup
    parser.add_argument("--amp", action='store_true', help="Enable Automatic Mixed Precision")
    
    # LoRA Hyperparameters (as per the original code)
    parser.add_argument("--lora", default=0, type=int, help="Enable LoRA")
    parser.add_argument("--lora_targets", type=str, default="image_enc_attn", help="LoRA targets")
    parser.add_argument("--lora_rank", default=4, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA dropout")

    # Augmentation Hyperparameters
    parser.add_argument("--horizontal_flip", default=0, type=int,  help="Enable horizontal flip augmentation")
    parser.add_argument("--vertical_flip", default=0, type=int,  help="Enable vertical flip augmentation")
    parser.add_argument("--random_rotate", default=0, type=int,  help="Enable random rotation augmentation")
    parser.add_argument("--elastic_transform", default=0, type=int, help="Enable elastic transform augmentation")
    parser.add_argument("--random_crop_size", default=None, help="Size (height, width) for random crop")
    parser.add_argument("--normalize", default=0, type=int,  help="Enable normalization of images")

    return parser
    
def plot_graph(loss_array, y_label, filename="loss_graph.png"):
    cma = np.cumsum(loss_array) / (np.arange(len(loss_array)) + 1)    
    plt.plot(loss_array, label="Curve", color="blue")    
    plt.plot(cma, label="CMA", color="orange", linestyle='--')    
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.title("Cumulative Moving Average vs Epochs")
    
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()



def main(args):
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda") 
    dataset = CustomDataset(args.dataset_name, train=True, args=args)

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
            lora_alpha=args.lora_alpha,  
            lora_dropout=args.lora_dropout,  
        )
        sam2_model = get_peft_model(sam2_model, lora_config)
        sam2_model.print_trainable_parameters()

    predictor = SAM2ImagePredictor(sam2_model)

    
    for name, param in predictor.model.named_parameters():
        if "sam_mask_decoder" in name or "sam_prompt_encoder" in name:
            param.requires_grad = True

    scores = {}
    ious = {}
    losses = []
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define optimizer
    if args.opt == "sgd":
        optimizer = SGD(
            params=predictor.model.parameters(),
            lr=args.lr,
            momentum=args.momentum if args.momentum > 0 else 0,  # Use momentum if specified
            weight_decay=args.weight_decay,
        )
    elif args.opt in ["adam", "adamw"]:
        OptimClass = Adam if args.opt == "adam" else AdamW
        optimizer = OptimClass(
            params=predictor.model.parameters(), lr=args.lr, betas=eval(args.opt_betas), weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.opt}")

    # AMP setup
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Scheduler setup
    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)
    elif args.sched == "step":
        scheduler = StepLR(optimizer, step_size=args.decay_epochs, gamma=args.decay_rate)
    elif args.sched == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=[args.decay_epochs], gamma=args.decay_rate)
    elif args.sched == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=args.patience_epochs, factor=args.decay_rate)
    else:
        scheduler = None


# Reference:
# https://towardsdatascience.com/train-fine-tune-segment-anything-2-sam-2-in-60-lines-of-code-928dd29a63b3

# Training loop
    start_time = time.time()
    itr = 0
    while itr < args.num_train_epochs:
        image, mask, input_point, input_label = dataset[np.random.randint(len(dataset))]
        if mask.shape == (0,):
            continue  
        predictor.set_image(image)

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]
        ]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )
        
        sorted_indices = torch.argsort(prd_scores, dim=1, descending=True)

        prd_masks = torch.gather(prd_masks, 1, sorted_indices.unsqueeze(-1).unsqueeze(-1).expand_as(prd_masks))
        prd_scores = torch.gather(prd_scores, 1, sorted_indices)

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])

        loss = (
            -gt_mask * torch.log(prd_mask + 1e-5)
            - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-5)
        ).mean()

        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.sched == "plateau":
            scheduler.step(loss)
        elif scheduler:
            scheduler.step()

        torch.save(
            predictor.model.state_dict(),
            os.path.join(args.output_dir, "sam2model.torch"),
        )

        # calculate iou

        prd_mask_bin = (prd_mask > 0.5).float()  

        intersection = (gt_mask * prd_mask_bin).sum(dim=(1, 2))  
        union = ((gt_mask + prd_mask_bin) > 0).float().sum(dim=(1, 2))  

        iou = intersection / (union + 1e-6)  
        
        score = np.mean(prd_scores[:, 0].cpu().detach().numpy())
        scores[f"epoch_{itr}_score"] = score
        ious[f"epoch_{itr}_iou"] = iou.mean().item()
        
        losses.append(loss.item())
        print(f"Epoch: {itr} Loss: {loss.item()} Acc: {iou.mean()}")

        itr += 1

    cost = time.time() - start_time
    mean_score = mean(scores.values())
    mean_iou = mean(ious.values())

    # plot_graph(losses, "Loss", "loss_graph.png")
   
    if args.return_scores_per_epoch:
        return mean_iou, ious, cost
    else:
        return mean_iou, cost


if __name__ == "__main__":
    parser = get_parser() 
    args = parser.parse_args()

    mean_score, cost = main(args)
    print("Epochs: {}, Score: {}, Cost: {}".format(args.num_train_epochs, mean_score, cost))
    


