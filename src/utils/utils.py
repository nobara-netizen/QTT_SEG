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

def get_config_space():
    cs = ConfigurationSpace("cv-segmentation")
    # Seed
    rs = OrdinalHyperparameter("seed", [1, 7, 42, 99, 123, 256, 512, 1024, 1337, 2024, 3000, 31415, 54321, 65536, 99999])
    # Learning rate and weight decay
    lr = OrdinalHyperparameter("learning_rate", [0.0001, 0.00001, 0.000001])
    wd = OrdinalHyperparameter("weight_decay", [1e-4, 1e-5])

    # Model selection
    model = Categorical("model_name", ["SAM1", "SAM2"])

    # LoRA hyperparameters
    lora = OrdinalHyperparameter("lora", [0, 1])  # Matches default setting
    lora_rank = OrdinalHyperparameter("lora_rank", [4, 8])
    lora_dropout = OrdinalHyperparameter("lora_dropout", [0.1, 0.2, 0.3])

    # Augmentation hyperparameters
    horizontal_flip = OrdinalHyperparameter("horizontal_flip", [0, 1])
    vertical_flip = OrdinalHyperparameter("vertical_flip", [0, 1])
    random_rotate = OrdinalHyperparameter("random_rotate", [0, 1])
    elastic_transform = OrdinalHyperparameter("elastic_transform", [0, 1])
    normalize = OrdinalHyperparameter("normalize", [0, 1])

    # Optimizer arguments
    opt = Categorical("opt", ["sgd", "adam", "adamw", "rmsprop"])
    momentum = OrdinalHyperparameter("momentum", [0.8, 0.9])
    opt_betas = Categorical("opt_betas", [(0.9, 0.999), (0.85, 0.995), (0.8, 0.9)])

    # Scheduler arguments
    sched = Categorical("sched", ["cosine", "onecycle", "plateau"])
    decay_epochs = OrdinalHyperparameter("decay_epochs", [30, 60, 90])
    decay_rate = OrdinalHyperparameter("decay_rate", [0.1, 0.3, 0.5])
    patience_epochs = OrdinalHyperparameter("patience_epochs", [5])

    # LoRA Targets - Conditional on model_name
    lora_targets_sam1 = Categorical("lora_targets_sam1", ["attn", "self_attn", "cross_attn"])
    lora_targets_sam2 = Categorical("lora_targets_sam2", ["image_enc_attn", "image_enc_mlp", "memory_self_attn", "memory_cross_attn"])

    # Add hyperparameters to configuration space
    cs.add_hyperparameters([
        rs, lr, wd, model, lora, lora_rank, lora_dropout,
        horizontal_flip, vertical_flip, random_rotate, elastic_transform, normalize,
        opt, momentum, opt_betas, sched, decay_epochs, decay_rate, patience_epochs,
        lora_targets_sam1, lora_targets_sam2
    ])

    # Conditional Dependencies
    cs.add_condition(EqualsCondition(lora_targets_sam1, model, "SAM1"))
    cs.add_condition(EqualsCondition(lora_targets_sam2, model, "SAM2"))

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
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False 

