import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import matplotlib.pyplot as plt
from typing import Dict, List
import random
import torch
import numpy as np




def plot_dice_coefficients(
    dice_per_class_per_epoch_train: Dict[str, List[float]], 
    dice_per_class_per_epoch_val: Dict[str, List[float]], 
    class_names: List[str],
    experiment_id:str,
    reports_dir
) -> None:
    """
    Plot Dice coefficients per class for both training and validation over epochs.
    Training lines will be solid and validation lines dashed. 

    Parameters:
    - dice_per_class_per_epoch_train (dict): Dictionary with training Dice coefficients per class.
    - dice_per_class_per_epoch_val (dict): Dictionary with validation Dice coefficients per class.
    - class_names (list): List of class names.
        """
    num_epochs = len(next(iter(dice_per_class_per_epoch_train.values())))
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 10))

    colors = plt.cm.tab10.colors
    line_styles = ['-', '--']

    for i, class_name in enumerate(class_names):
        plt.plot(epochs, dice_per_class_per_epoch_train[class_name], 
                 label=f'Train Dice for {class_name}', color=colors[i % len(colors)], linestyle=line_styles[0])
        plt.plot(epochs, dice_per_class_per_epoch_val[class_name], 
                 label=f'Val Dice for {class_name}', color=colors[i % len(colors)], linestyle=line_styles[1])
    
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title(f'{experiment_id} - Dice Coefficient per Class for Train and Validation over Epochs')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(reports_dir, f'{experiment_id}_dice_coefficients_train_val.png'))
    plt.show()    

def plot_learning_curve(
    train_loss_history: Dict[str, List[float]], 
    val_loss_history: Dict[str, List[float]], 
    class_names: List[str],
    experiment_id : str,
     reports_dir:str ) -> None:
    """
    Plot the learning curve for training and validation loss over epochs for each class.

    Args:
        train_loss_history (dict): Dictionary of average training loss values per epoch for each class.
        val_loss_history (dict): Dictionary of average validation loss values per epoch for each class.
        class_names (list): List of class names.
    """
    num_epochs = len(next(iter(train_loss_history.values())))
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 10))

    
    colors = plt.cm.tab10.colors  # Use a colormap with distinct colors
    line_styles = {
        'train': '-',
        'val': '--'
    }

    for i, class_name in enumerate(class_names):
        color = colors[i % len(colors)]  # Cycle through colors
        linestyle_train = line_styles['train']
        linestyle_val = line_styles['val']
        
        
        plt.plot(epochs, train_loss_history.get(class_name, [float('nan')] * num_epochs), 
                 label=f'Train Loss for {class_name}', 
                 color=color, linestyle=linestyle_train, marker='o')

        
        plt.plot(epochs, val_loss_history.get(class_name, [float('nan')] * num_epochs), 
                 label=f'Val Loss for {class_name}', 
                 color=color, linestyle=linestyle_val, marker='x')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{experiment_id} - Training and Validation Loss per Class over Epochs')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(reports_dir,f'{experiment_id}_learning_curve_loss.png'))
    plt.show()




def visualize_predictions(model, val_dataset, device, class_labels, number_of_samples, reports_dir, experiment_id):
    """
    Visualizes the predicted and true masks for a random sample of images from the validation dataset.
    
    Args:
    - model: Trained model to make predictions.
    - val_dataset: Dataset containing images and true masks.
    - device: Device on which the model is running (e.g., 'cuda' or 'cpu').
    - class_labels: Dictionary containing class labels and their active state.
    - number_of_samples: Number of random samples to visualize.
    - reports_dir: Directory to save the visualized prediction images.
    - experiment_id: Unique experiment identifier for saving the plots.
    """
    model.eval()
    with torch.no_grad():
        
        sample_indices = random.sample(range(len(val_dataset)), number_of_samples)

        for idx in sample_indices:
            cropped_image, cropped_masks = val_dataset[idx]

            if isinstance(cropped_image, np.ndarray):
                cropped_image = torch.from_numpy(cropped_image).float()
            cropped_image = cropped_image.to(device).unsqueeze(0)

            outputs = model(cropped_image)
            pred_masks = (outputs > 0.5).float()

            cropped_image = cropped_image.cpu().squeeze().permute(1, 2, 0).numpy()

            num_true_masks = len(cropped_masks)
            num_pred_masks = pred_masks.shape[1]  
            total_plots = num_true_masks + num_pred_masks + 1

            plt.figure(figsize=(15, 5 * (total_plots // 3 + 1)))

            plt.subplot(total_plots // 3 + 1, 3, 1)
            plt.imshow(cropped_image)
            plt.title("Image")

            for i, (key, mask) in enumerate(cropped_masks.items()):
                plt.subplot(total_plots // 3 + 1, 3, i + 2)
                plt.imshow(mask.cpu().squeeze().numpy(), cmap="gray")
                plt.title(f"True Mask: {key}")

            active_classes = [label for i, (label, is_active) in enumerate(class_labels.items()) if is_active]

            for i, label in enumerate(active_classes):
                if i < num_pred_masks:
                    mask = pred_masks[0, i].cpu().numpy()
                    plt.subplot(total_plots // 3 + 1, 3, num_true_masks + 2 + i)
                    plt.imshow(mask, cmap="gray")
                    plt.title(f"Predicted Mask: {label}")

            plt.tight_layout()
            plt.savefig(os.path.join(reports_dir, f"{experiment_id}_predicted_masks_sampleID_{idx}.png"))
            plt.show()
