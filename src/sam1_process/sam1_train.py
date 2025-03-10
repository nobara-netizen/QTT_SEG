import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import SamProcessor, SamModel
from tqdm import tqdm
from statistics import mean
import time
from torch.nn.functional import threshold, normalize
from src.data.custom_dataloader import CustomDataset
import argparse
import os
from peft import LoraConfig, get_peft_model
import random
from src.utils.utils import set_seed, plot_graph

TARGET_MODULES_DICT = {
    "attn": [
        "attn.qkv",
        "attn.proj",
        "attn.qkv",
        "attn.proj"
    ],
    "self_attn": [
        "self_attn.q_proj", 
        "self_attn.k_proj", 
        "self_attn.v_proj", 
        "self_attn.out_proj"
    ],
    "cross_attn": [
        "cross_attn_token_to_image.q_proj", "cross_attn_token_to_image.k_proj",
        "cross_attn_token_to_image.v_proj", "cross_attn_token_to_image.out_proj",
        "cross_attn_image_to_token.q_proj", "cross_attn_image_to_token.k_proj",
        "cross_attn_image_to_token.v_proj", "cross_attn_image_to_token.out_proj",
    ]
}

def get_parser():
    parser = argparse.ArgumentParser()

    # Fixed args
    parser.add_argument("--dataset_name", default="semantic-drone-dataset", type=str, help="Dataset Name")
    parser.add_argument("--seed", default=0, type=int, help="Seed for sampling")
    parser.add_argument("--output_dir", default="./outputs", type=str, help="Output path")
    parser.add_argument("--num_train_epochs", default=2, type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")

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
    parser.add_argument("--lora", default=1, type=int, choices=[0, 1], help="Enable LoRA")
    parser.add_argument("--lora_targets",type=str,choices=TARGET_MODULES_DICT.keys(),default="cross_attn",help="Choose which attention modules to apply LoRA to: 'attn', 'self_attn', or 'cross_attn'.")
    parser.add_argument("--lora_rank", default=4, type=int, choices=(4, 8), help="LoRA rank")
    parser.add_argument("--lora_dropout", default=0.1, type=float, choices=(0.1,0.2, 0.3), help="LoRA dropout")

    # Augmentation Hyperparameters
    parser.add_argument("--horizontal_flip", default=0, type=int, choices=[0, 1], help="Enable horizontal flip augmentation")
    parser.add_argument("--vertical_flip", default=0, type=int, choices=[0, 1], help="Enable vertical flip augmentation")
    parser.add_argument("--random_rotate", default=0, type=int, choices=[0, 1], help="Enable random rotation augmentation")
    parser.add_argument("--elastic_transform", default=0, type=int, choices=[0, 1], help="Enable elastic transform augmentation")
    parser.add_argument("--normalize", default=0, type=int, choices=[0, 1], help="Enable normalization of images")

    return parser

def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """
    Compute IoU for binary segmentation masks.

    Args:
        pred_mask (torch.Tensor): Predicted mask (logits or probabilities), shape (batch, 1, H, W).
        gt_mask (torch.Tensor): Ground truth mask (binary values), shape (batch, 1, H, W).
        threshold (float): Threshold to binarize predicted mask.

    Returns:
        torch.Tensor: Mean IoU across the batch.
    """
    # Binarize the predicted mask
    pred_mask = (pred_mask > threshold).float()
    gt_mask = gt_mask.float()

    # Remove the channel dimension (1) to get (batch, H, W)
    pred_mask = pred_mask.squeeze(1)  
    gt_mask = gt_mask.squeeze(1)

    # Compute intersection and union
    intersection = (pred_mask * gt_mask).sum(dim=(1, 2))  # Sum over H and W
    union = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2)) - intersection

    # Compute IoU
    iou = intersection / (union + 1e-6)  # Shape: (batch,)

    return iou.mean().item()


def get_bounding_box(masks):
  # get bounding box from mask
    bounding_boxes = []
    for i in range(masks.shape[0]):  # Loop over the 10 masks
        ground_truth_map = masks[i]  # Shape (256, 256)
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]
        bounding_boxes.append(bbox)
    return bounding_boxes
     

class SAMDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)
    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[prompt], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs


def train(args):
    torch.backends.cudnn.benchmark = True  # Optimize GPU efficiency

    set_seed(args.seed)
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Limit dataset size to avoid memory issues
    dataset = CustomDataset(args.dataset_name, train=True, sam1=True)
    sampled_indices = torch.randperm(len(dataset))[:100]  # Use only 100 samples for training
    sampled_dataset = Subset(dataset, sampled_indices)
    train_dataset = SAMDataset(dataset=sampled_dataset, processor=processor)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # Load Model
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    
    # Apply LoRA if enabled
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            target_modules=TARGET_MODULES_DICT[args.lora_targets],
            lora_dropout=args.lora_dropout,
        )
        model = get_peft_model(model, lora_config)

    # Move model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Define Optimizer
    opt_betas = eval(args.opt_betas) if isinstance(args.opt_betas, str) else args.opt_betas
    optimizers = {
        "sgd": lambda: SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay),
        "adam": lambda: Adam(model.parameters(), lr=args.lr, betas=opt_betas, weight_decay=args.weight_decay),
        "adamw": lambda: AdamW(model.parameters(), lr=args.lr, betas=opt_betas, weight_decay=args.weight_decay),
        "rmsprop": lambda: RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay),
    }
    optimizer = optimizers.get(args.opt, lambda: ValueError(f"Unsupported optimizer: {args.opt}"))()

    # Enable Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # Define Scheduler
    scheduler = None
    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)
    elif args.sched == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=args.patience_epochs, factor=args.decay_rate)
    elif args.sched == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.num_train_epochs, steps_per_epoch=len(train_dataloader))

    # Training Loop
    model.train()
    torch.cuda.empty_cache()  # Free up GPU memory before starting
    start_time = time.time()

    losses = []
    iou_scores = {}

    for epoch in range(args.num_train_epochs):
        epoch_losses = []
        epoch_ious = []

        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_boxes=batch["input_boxes"].to(device),
                    multimask_output=False,
                )

                predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(2))
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)

                # Compute Loss
                loss = nn.BCEWithLogitsLoss()(predicted_masks, ground_truth_masks)

            # Backpropagation with AMP
            scaler.scale(loss).backward(retain_graph=False)  # Avoid retaining graph
            scaler.step(optimizer)
            scaler.update()

            # Step scheduler
            if args.sched == "plateau":
                scheduler.step(loss) if scheduler else None
            elif scheduler:
                scheduler.step()

            epoch_losses.append(loss.item())

            # Compute IoU and move tensors to CPU to free memory
            iou = calculate_iou(predicted_masks.cpu(), ground_truth_masks.cpu())
            epoch_ious.append(iou)

        # Compute mean loss and IoU for the epoch
        mean_loss = np.mean(epoch_losses)
        mean_iou = np.mean(epoch_ious)

        losses.append(mean_loss)
        iou_scores[f"epoch_{epoch}_iou"] = mean_iou

        print(f"Epoch {epoch}: Loss: {mean_loss:.4f}, IoU: {mean_iou:.4f}")

    # Training Time
    cost = time.time() - start_time

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    if args.return_scores_per_epoch:
        return {"dataset": args.dataset_name, "score": np.mean(list(iou_scores.values())), "cost": cost}, iou_scores

    return {"dataset": args.dataset_name, "score": np.mean(list(iou_scores.values())), "cost": cost}

def test(dataset, args, model_path,zero_shot=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    dataset = CustomDataset(args.dataset_name, train=False, sam1=True)
    test_dataset = SAMDataset(dataset=dataset, processor=processor)
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    

    if not zero_shot:
        if args.lora:
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=16,
                target_modules=TARGET_MODULES_DICT[args.lora_targets],  # Apply dynamically selected modules
                lora_dropout=args.lora_dropout,
            )
            model = get_peft_model(model, lora_config)
        model.load_state_dict(torch.load(os.path.join(args["output_dir"], "sam1model.torch")))
    model.to(device)
    model.eval()
    
    # Prepare test dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)
    
    iou_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_boxes": batch["input_boxes"].to(device),
            }
            outputs = model(**inputs, multimask_output=False)
            predicted_masks = torch.sigmoid(outputs.pred_masks.squeeze(2)).to(device)
        
            # Compute IoU
            ground_truth_masks = batch["ground_truth_mask"].to(device)
            iou = calculate_iou(predicted_masks, ground_truth_masks)
            iou_scores.append(iou)
    
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU: {mean_iou:.4f}")
    
    return mean_iou
        
     

if __name__ == "__main__":
    parser = get_parser() 
    args = parser.parse_args()
    report = train(args)
    # test_report = test(dataset, args.__dict__, model_path="./outputs")

