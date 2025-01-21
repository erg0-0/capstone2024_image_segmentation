import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Union



def tensor_to_image(tensor: Union[torch.Tensor, Image.Image]) -> np.ndarray:
    """
    Converts a tensor or PIL image to a NumPy array representation of an image.

    This function checks the type of the input. If the input is a PyTorch tensor,
    it converts it to a NumPy array, rearranging the dimensions if necessary. 
    If the input is a PIL Image, it converts it to a NumPy array and normalizes the pixel values
    to the range [0, 1]. The function handles both 3D and 4D tensors appropriately.

    Parameters:
        tensor (Union[torch.Tensor, Image.Image]): The input image tensor or PIL image 
                                                    to be converted.

    Returns:
        np.ndarray: A NumPy array representation of the image with pixel values in the 
                    range [0, 1].

    Raises:
        TypeError: If the input is neither a PIL Image nor a PyTorch tensor.
    """
    if isinstance(tensor, torch.Tensor):
        np_image = tensor.cpu().numpy()
        if np_image.ndim == 3 and np_image.shape[0] == 3:
            np_image = np.transpose(np_image, (1, 2, 0))
        elif np_image.ndim == 4:
            np_image = np_image.squeeze(0)
            if np_image.ndim == 3 and np_image.shape[0] == 3:
                np_image = np.transpose(np_image, (1, 2, 0))
        np_image = np.clip(np_image, 0, 1)
        return np_image
    elif isinstance(tensor, Image.Image):
        return np.array(tensor) / 255.0
    else:
        raise TypeError("Input must be a PIL Image or a torch Tensor.")

def show_random_sample(dataset: Dataset, reports_dir:str, experiment_id: str ) -> None:
    """
    Displays and saves random samples from a dataset, including images and their associated masks.

    This function selects three random samples from the provided dataset and visualizes 
    each image alongside its corresponding masks. The images are converted from tensor 
    format to NumPy arrays for display. The original image of each sample is also saved 
    as a JPEG file in the specified reports directory.

    Parameters:
        dataset (Dataset): The dataset from which to sample images and masks.
        reports_dir (str): The directory where the sampled images will be saved.
        experiment_id (str): An identifier for the experiment, used in the saved file names.

    Returns:
        None: This function does not return any value; it only displays and saves images.
    """
    with torch.no_grad():
        sample_indices = random.sample(range(len(dataset)), 3)
        for idx in sample_indices:
            cropped_image, cropped_masks = dataset[idx]

            image_np = tensor_to_image(cropped_image)
            masks_np = {key: tensor_to_image(mask) for key, mask in cropped_masks.items()}

            plt.figure(figsize=(20, 10))

            plt.subplot(1, len(masks_np) + 1, 1)
            plt.imshow(image_np)
            plt.title(f"Image sample {idx}", fontsize = 15)

            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
            
            image_pil.save(os.path.join(reports_dir, f"{experiment_id}_sampleID_{idx}_image.jpg"))

            for i, (key, mask) in enumerate(masks_np.items(), 2):
                plt.subplot(1, len(masks_np) + 1, i)
                if mask.ndim == 2:
                    plt.imshow(mask, cmap='gray')
                elif mask.ndim == 3 and mask.shape[2] == 3:
                    plt.imshow(mask)
                elif mask.ndim == 3 and mask.shape[0] == 1:
                    plt.imshow(mask.squeeze(0), cmap='gray')
                else:
                    raise ValueError(f"Unsupported mask shape: {mask.shape}")
                plt.title(key,fontsize = 15)

            plt.tight_layout()
            plt.show()