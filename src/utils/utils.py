import argparse
from ConfigSpace import (
    Categorical,
    ConfigurationSpace,
    Constant,
    EqualsCondition,
    OrConjunction,
    OrdinalHyperparameter,
)
import random
import numpy as np
import torch

TARGET_MODULES_DICT = {
    "modules" :["sam_mask_decoder.transformer.layers.0.self_attn.q_proj",
    "sam_mask_decoder.transformer.layers.0.self_attn.v_proj",
    "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj",
    "sam_mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj",
    "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj",
    "sam_mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj",
    "sam_mask_decoder.transformer.layers.1.self_attn.q_proj",
    "sam_mask_decoder.transformer.layers.1.self_attn.v_proj",
    "sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj",
    "sam_mask_decoder.transformer.layers.1.cross_attn_token_to_image.v_proj",
    "sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.q_proj",
    "sam_mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj",
    "sam_mask_decoder.transformer.final_attn_token_to_image.q_proj",
    "sam_mask_decoder.transformer.final_attn_token_to_image.v_proj"]
}

# TARGET_MODULES_DICT = {
#     "modules" : [
#     "mask_decoder.transformer.layers.0.self_attn.q_proj",
#     "mask_decoder.transformer.layers.0.self_attn.v_proj",
#     "mask_decoder.transformer.layers.0.cross_attn_token_to_image.q_proj",
#     "mask_decoder.transformer.layers.0.cross_attn_token_to_image.v_proj",
#     "mask_decoder.transformer.layers.0.cross_attn_image_to_token.q_proj",
#     "mask_decoder.transformer.layers.0.cross_attn_image_to_token.v_proj",
#     "mask_decoder.transformer.layers.1.self_attn.q_proj",
#     "mask_decoder.transformer.layers.1.self_attn.v_proj",
#     "mask_decoder.transformer.layers.1.cross_attn_token_to_image.q_proj",
#     "mask_decoder.transformer.layers.1.cross_attn_token_to_image.v_proj",
#     "mask_decoder.transformer.layers.1.cross_attn_image_to_token.q_proj",
#     "mask_decoder.transformer.layers.1.cross_attn_image_to_token.v_proj",
#     "mask_decoder.transformer.final_attn_token_to_image.q_proj",
#     "mask_decoder.transformer.final_attn_token_to_image.v_proj",
#     ]
# }


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="building", type=str, help="Dataset Name")
    parser.add_argument("--seed", default=0, type=int, help="Seed for sampling")
    parser.add_argument("--output_dir", default="./outputs", type=str, help="Output path")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--num_prompts", default=64, type=int, help="Number of training prompts")
    parser.add_argument('--return_scores_per_epoch', action='store_true', help="Return scores per epoch")

    # Hyperparameters
    parser.add_argument("--lr", default=1e-4, type=float, 
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", default=1e-5, type=float, 
                        help="Weight decay (L2 regularization)")

    # LoRA Hyperparameters
    parser.add_argument("--lora", default=1, type=int, choices=[0, 1], help="Enable LoRA")
    parser.add_argument("--lora_targets",type=str,default="modules",help="Choose which attention modules to apply LoRA to: 'self_attn', or 'cross_attn'.")
    parser.add_argument("--lora_rank", default=8, type=int, help="LoRA rank")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA Dropout")

    # Optimizer arguments
    parser.add_argument("--opt", default="adam", help="Optimizer type")
    parser.add_argument("--opt_betas", type=eval, default=(0.9, 0.999), 
                        help="Betas for Adam/AdamW (Tuple)")

    # Scheduler arguments
    parser.add_argument("--sched", choices=["cosine", "plateau","onecycle"], 
                        default="cosine", help="Learning rate scheduler")
    parser.add_argument("--decay_rate", default=0.1, type=float, 
                        help="Decay rate")
    parser.add_argument("--patience_epochs", default=1, type=int, help="Patience epochs for plateau scheduler")

    # Augmentation Hyperparameters
    parser.add_argument("--horizontal_flip", default=0, type=int, choices=[0, 1], help="Enable horizontal flip augmentation")
    parser.add_argument("--vertical_flip", default=0, type=int, choices=[0, 1], help="Enable vertical flip augmentation")
    parser.add_argument("--random_rotate", default=0, type=int, choices=[0, 1], help="Enable random rotation augmentation")

    return parser


def get_config_space():
    cs = ConfigurationSpace("cv-segmentation")
    # Seed
    # Seed: include a few stable and reproducible ones
    rs = OrdinalHyperparameter("seed", [0, 42, 1337])

    # Prompts: fine-grained range that covers typical usage
    ps = OrdinalHyperparameter("num_prompts", [64, 128, 256, 512])

    # Learning rate: narrower, empirically strong for fine-tuning transformers
    lr = OrdinalHyperparameter("lr", [5e-5, 1e-4, 3e-4,1e-3])

    # Weight decay: use small values to regularize without killing performance
    wd = OrdinalHyperparameter("weight_decay", [0.0, 1e-5, 5e-5, 1e-4])

    # Model choices
    model = Categorical("model_name", ["SAM2"])

    # LoRA: enabled with sensible defaults
    lora = OrdinalHyperparameter("lora", [0,1])
    lora_rank = OrdinalHyperparameter("lora_rank", [8, 16, 32])
    lora_dropout = OrdinalHyperparameter("lora_dropout", [0.0, 0.1])
    lora_targets = Categorical("lora_targets", ["modules"])

    # Augmentations: usually helpful, especially on small datasets
    horizontal_flip = OrdinalHyperparameter("horizontal_flip", [0, 1])
    vertical_flip = OrdinalHyperparameter("vertical_flip", [0, 1])
    random_rotate = OrdinalHyperparameter("random_rotate", [0, 1])

    # Optimizer: AdamW is best for transformers, rest as fallbacks
    opt = Categorical("opt", ["adamw", "adam"])
    opt_betas = Categorical("opt_betas", [(0.9, 0.999)])

    # Scheduler: cosine and onecycle are well-tested for segmentation
    sched = Categorical("sched", ["cosine", "onecycle", "plateau"])
    decay_rate = OrdinalHyperparameter("decay_rate", [0.1])
    patience_epochs = OrdinalHyperparameter("patience_epochs", [5, 3])

    # Add all hyperparameters to config space
    cs.add_hyperparameters([
        rs, ps, lr, wd, model, lora, lora_rank, lora_dropout, lora_targets,
        horizontal_flip, vertical_flip, random_rotate,
        opt, opt_betas, sched, decay_rate, patience_epochs,
    ])

    return cs


    # Add hyperparameters to configuration space
    cs.add_hyperparameters([
        rs, ps, lr, wd, model, lora, lora_rank, lora_dropout,lora_targets,
        horizontal_flip, vertical_flip, random_rotate,
        opt, opt_betas, sched, decay_epochs, decay_rate, patience_epochs,
    ])

    return cs

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

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

