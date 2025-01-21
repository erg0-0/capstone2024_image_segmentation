import random
from PIL import ImageOps, Image
from typing import Tuple, Dict
import torch
from torch.utils.data import DataLoader, Dataset

def split_dataset(dataset, split_ratio, batch_size, shuffle=True):
    """
    Splits a given dataset into two subsets based on a specified ratio and returns data loaders for both subsets,
    along with the sizes of the split datasets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        split_ratio (float): The ratio of the dataset to allocate to the first split. The remainder goes to the second split.
        batch_size (int): The batch size to use for the DataLoader for both splits.
        shuffle (bool, optional): Whether to shuffle the dataset for the first DataLoader. Defaults to True.

    Returns:
        DataLoader: DataLoader for the first split of the dataset.
        DataLoader: DataLoader for the second split of the dataset.
        int: The size of the first split.
        int: The size of the second split.
    """
    split_size = int(split_ratio * len(dataset))
    remainder_size = len(dataset) - split_size
    split_dataset, remainder_dataset = torch.utils.data.random_split(dataset, [split_size, remainder_size])
    split_loader = DataLoader(split_dataset, batch_size=batch_size, shuffle=shuffle)
    remainder_loader = DataLoader(remainder_dataset, batch_size=batch_size, shuffle=False)
    return split_loader, remainder_loader, split_size, remainder_size



def pad_image_if_needed(image: Image.Image, crop_size: Tuple[int,int])->Image.Image:
    """
    Pads an image to ensure it meets the specified crop size.

    If the dimensions of the input image are smaller than the specified crop size,
    this function will add padding to the right and/or bottom of the image.
    The padding is filled with black (0 value).

    Parameters:
        image (Image.Image): The input image to be padded.
        crop_size (Tuple[int, int]): A tuple specifying the desired width and height
                                      for the image in pixels (crop_width, crop_height).

    Returns:
        Image.Image: The padded image with dimensions equal to or greater than the 
                      specified crop size.
    """
    width, height = image.size
    crop_width, crop_height = crop_size
    
    pad_width = max(0, crop_width - width)
    pad_height = max(0, crop_height - height)
    
    if pad_width > 0 or pad_height > 0:
        padding = (0, 0, pad_width, pad_height)
        image = ImageOps.expand(image, padding, fill=0)
    
    return image

def get_random_crop_coords(image_size: Tuple[int,int], crop_size: Tuple[int,int])->Tuple [int,int,int,int]:
    """
    Generates random coordinates for cropping an image.

    This function calculates a random rectangular region within an image
    specified by its dimensions, ensuring that the crop size fits within
    the image bounds. If the crop size is larger than the image dimensions,
    the coordinates will default to (0, 0) for the top-left corner.

    Parameters:
        image_size (Tuple[int, int]): A tuple representing the width and height
                                       of the input image (width, height).
        crop_size (Tuple[int, int]): A tuple specifying the desired width and height
                                      of the crop (crop_width, crop_height).

    Returns:
        Tuple[int, int, int, int]: A tuple containing the coordinates for cropping 
                                    the image in the format (left, top, right, bottom).
    """
    width, height = image_size
    crop_width, crop_height = crop_size
    
    left = random.randint(0, width - crop_width) if width > crop_width else 0
    top = random.randint(0, height - crop_height) if height > crop_height else 0
    right = left + crop_width
    bottom = top + crop_height
    
    return (left, top, right, bottom)

def crop_image(image: Image.Image, crop_coords:Tuple[int,int,int,int])-> Image.Image:
    """
    Crops an image to the specified coordinates.

    This function takes an image and a tuple of coordinates that define the area to be 
    cropped. The coordinates are expected in the format (left, top, right, bottom).

    Parameters:
        image (Image.Image): The input image to be cropped.
        crop_coords (Tuple[int, int, int, int]): A tuple containing the coordinates for 
                                                  cropping in the format (left, top, 
                                                  right, bottom).

    Returns:
        Image.Image: The cropped image based on the provided coordinates.
    """
    return image.crop(crop_coords)

def crop_image_and_masks(image: Image.Image, masks:Dict[str,Image.Image], crop_coords: Tuple[int,int,int,int])->Tuple[Image.Image, Dict[str,Image.Image]]:
    """
    Crops an image and its associated masks to the specified coordinates.

    This function first pads the input image and masks to ensure they meet the desired 
    crop dimensions, then crops them based on the provided coordinates. The function 
    returns the cropped image and a dictionary of cropped masks, both resized to the 
    specified crop size.

    Parameters:
        image (Image.Image): The input image to be cropped.
        masks (Dict[str, Image.Image]): A dictionary where keys are mask identifiers 
                                         and values are the corresponding mask images.
        crop_coords (Tuple[int, int, int, int]): A tuple containing the coordinates for 
                                                  cropping in the format (left, top, 
                                                  right, bottom).

    Returns:
        Tuple[Image.Image, Dict[str, Image.Image]]: A tuple containing the cropped image 
                                                     and a dictionary of cropped masks, 
                                                     both resized to the crop size.
    """
    crop_width = crop_coords[2] - crop_coords[0]
    crop_height = crop_coords[3] - crop_coords[1]
    
    padded_image = pad_image_if_needed(image, (crop_width, crop_height))
    padded_masks = {key: pad_image_if_needed(mask, (crop_width, crop_height)) for key, mask in masks.items()}
    
    cropped_image = crop_image(padded_image, crop_coords)
    cropped_masks = {key: crop_image(mask, crop_coords) for key, mask in padded_masks.items()}
    
    cropped_image = cropped_image.resize((crop_width, crop_height))
    cropped_masks = {key: mask.resize((crop_width, crop_height)) for key, mask in cropped_masks.items()}
    
    return cropped_image, cropped_masks