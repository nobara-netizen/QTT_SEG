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
import matplotlib.pyplot as plt

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="danish-golf-courses-orthophotos", type=str, help="Dataset Name")
    parser.add_argument("--output_dir", default="./outputs", type=str, help="Output path")
    parser.add_argument("--num_train_epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument('--return_scores_per_epoch', action='store_true', help="Return scores per epoch")

    # Hyperparameters
    parser.add_argument("--lr", default=0.0001, type=float, help="Initial learning rate")
    parser.add_argument("--weight_decay", default=0.0001, type=float, help="Weight decay (L2 regularization)")

    # LoRA Hyperparameters
    parser.add_argument("--lora", default=0, type=int, choices=[0, 1], help="Enable LoRA")
    parser.add_argument("--lora_rank", default=4, type=int, help="LoRA rank")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA Dropout")

    # Scheduler arguments
    parser.add_argument("--sched", choices=["cosine", "cosine_warm", "step", "plateau", "onecycle", "poly"], 
                        default="cosine", help="Learning rate scheduler")
    parser.add_argument("--decay_rate", default=0.8, type=float, help="Decay rate (e.g., for plateau/step schedulers)")
    parser.add_argument("--patience_epochs", default=2, type=int, help="Patience epochs for plateau scheduler")

    # CosineAnnealingWarmRestarts specific
    parser.add_argument("--cosine_t0", default=2, type=int, help="T_0 for CosineAnnealingWarmRestarts")
    parser.add_argument("--cosine_t_mult", default=2, type=int, help="T_mult for CosineAnnealingWarmRestarts")

    # OneCycleLR specific
    parser.add_argument("--onecycle_pct_start", default=0.08, type=float, help="pct_start for OneCycleLR")
    parser.add_argument("--onecycle_div_factor", default=65, type=float, help="div_factor for OneCycleLR")
    parser.add_argument("--onecycle_final_div_factor", default=320, type=float, help="final_div_factor for OneCycleLR")

    # StepLR specific
    parser.add_argument("--step_size", default=3, type=int, help="Step size for StepLR")

    # PolynomialLR specific
    parser.add_argument("--poly_power", default=0.5, type=float, help="Power for PolynomialLR")

    # Augmentation Hyperparameters
    parser.add_argument("--horizontal_flip", default=0, type=int, choices=[0, 1], help="Enable horizontal flip augmentation")
    parser.add_argument("--vertical_flip", default=0, type=int, choices=[0, 1], help="Enable vertical flip augmentation")
    parser.add_argument("--random_rotate", default=1, type=int, choices=[0, 1], help="Enable random rotation augmentation")

    return parser


def get_config_space():
    cs = ConfigurationSpace("cv-segmentation")

    # Learning rate
    lr = OrdinalHyperparameter("lr", [
    1e-05, 1.2e-05, 1.5e-05, 2e-05, 2.5e-05, 3.5e-05, 5e-05, 6e-05, 6.5e-05,
    0.0001, 0.00012, 0.00018, 0.00025, 0.00032, 0.0004, 0.00048, 0.0005, 0.00055,
    0.0008, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007
])

    # Weight decay
    wd = OrdinalHyperparameter("weight_decay", [0.0, 1e-5, 5e-5, 1e-4])

    # LoRA
    lora = OrdinalHyperparameter("lora", [0, 1])
    lora_rank = OrdinalHyperparameter("lora_rank", [4, 8, 16])
    lora_dropout = OrdinalHyperparameter("lora_dropout", [0.0, 0.1])

    # Augmentations
    horizontal_flip = OrdinalHyperparameter("horizontal_flip", [0, 1])
    vertical_flip = OrdinalHyperparameter("vertical_flip", [0, 1])
    random_rotate = OrdinalHyperparameter("random_rotate", [0, 1])

    # Scheduler
    sched = Categorical("sched", ["cosine", "onecycle", "plateau", "cosine_warm", "step", "poly"])
    decay_rate = OrdinalHyperparameter("decay_rate", [0.1, 0.5, 0.8])
    patience_epochs = OrdinalHyperparameter("patience_epochs", [0, 1, 2])

    # --- Additional scheduler-specific hyperparameters ---

    # CosineWarmRestarts
    cosine_t0 = OrdinalHyperparameter("cosine_t0", [2, 3, 5])
    cosine_t_mult = OrdinalHyperparameter("cosine_t_mult", [1, 2])

    # OneCycleLR
    onecycle_pct_start = OrdinalHyperparameter("onecycle_pct_start",[0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100])
    onecycle_div_factor = OrdinalHyperparameter("onecycle_div_factor", [10, 15, 20, 25, 30, 40, 50, 65, 80, 100])
    onecycle_final_div_factor = OrdinalHyperparameter("onecycle_final_div_factor", [10, 20, 40, 80, 160, 320, 640, 1000])

    # StepLR
    step_size = OrdinalHyperparameter("step_size", [3, 5])

    # PolynomialLR
    poly_power = OrdinalHyperparameter("poly_power",[0.9, 0.5, 1.0])

    # Add all hyperparameters to config space
    cs.add_hyperparameters([
        lr, wd,
        lora, lora_rank, lora_dropout,
        horizontal_flip, vertical_flip, random_rotate,
        sched, decay_rate, patience_epochs,
        cosine_t0, cosine_t_mult,
        onecycle_pct_start, onecycle_div_factor, onecycle_final_div_factor,
        step_size, poly_power
    ])

    return cs

def plot_training_metrics(train_loss, train_iou, test_iou, save_path='training_metrics.png'):
    """
    Plots training loss, training IOU, and test IOU over epochs, and saves the plot to a file.
    
    Args:
        train_loss (list of float): Training loss per epoch
        train_iou (list of float): Training IOU per epoch
        test_iou (list of float): Test IOU per epoch
        save_path (str): Path to save the output plot image
    """
    epochs = range(1, len(train_loss) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training loss on the left y-axis
    ax1.plot(epochs, train_loss, 'r-', label='Train Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    # Create a second y-axis for IOU
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_iou, 'g--', label='Train IOU')
    ax2.plot(epochs, test_iou, 'b-.', label='Test IOU')
    ax2.set_ylabel('IOU', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')

    plt.title('Training Metrics Over Epochs')
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  

