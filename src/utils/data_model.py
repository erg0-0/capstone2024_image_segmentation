import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))
sys.path.append(project_root)
import torch
import numpy as np


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> float:
    """
    Calculate the Dice coefficient between predictions and target.
    This version assumes that predictions are probabilities and targets are binary masks.
    """

    if pred.dim() == 4 and pred.shape[1] == 1:  
        pred = torch.softmax(pred,dim=1)
    else:
        pred = torch.sigmoid(pred)
    
    pred_binary = (pred > 0.5).float()    
    pred_binary = pred_binary.contiguous().view(-1)
    
    target = target.contiguous().view(-1)
    
    intersection = (pred_binary * target).sum()
    return (2. * intersection + epsilon) / (pred_binary.sum() + target.sum() + epsilon)


def calculate_averages_per_class(data, metric_name, epochs):
    """Calculates the average per class over all epochs."""
    averages = {}
    for class_name, values in data.items():
        avg_value = sum(values) / epochs
        variable_name = f"{metric_name}_class_{class_name}_average"
        averages[variable_name] = avg_value
    return averages

def print_averages(averages, metric_type):
    """Prints class-wise averages."""
    print(f"===================={metric_type.upper()}====================")
    for variable_name, avg_value in averages.items():
        print(f"{variable_name} = {avg_value:.4f}")

def calculate_overall_average(data):
    """Calculates the overall average across all classes and epochs."""
    flat_data = [num for sublist in data.values() for num in sublist]
    return sum(flat_data) / len(flat_data)

def print_summary(phase, total_time, class_loss_averages, class_dice_averages, overall_loss):
    """Prints the summary for each phase (train/validation)."""
    print(f"===================={phase.upper()}====================")
    print_averages(class_loss_averages, f"{phase} loss")
    print_averages(class_dice_averages, f"{phase} dice")
    print(f"Overall {phase} average loss: {overall_loss:.4f}")
    print(f"Total {phase.capitalize()} Time: {total_time:.2f} sec")

def calculate_dice_for_report(enabled_masks, dice_data,all_classes):
    """Calculates the average Dice score per class."""
    return {class_name: np.mean(dice_data.get(class_name, [0.0])) if class_name in enabled_masks else 0.0
            for class_name in all_classes}

def calculate_loss_for_report(loss_data,all_classes):
    """Calculates the average loss per class."""
    return {class_name: np.mean(loss_data.get(class_name, [0.0])) 
            for class_name in all_classes}
