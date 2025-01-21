import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm


def find_capstone_dir(base_dir):
    while not base_dir.endswith('capstone_group5'):
        parent_dir = os.path.dirname(base_dir)
        if parent_dir == base_dir:  
            raise FileNotFoundError("Could not find 'capstone_group5' in the directory tree.")
        base_dir = parent_dir
    return base_dir

base_dir = os.getcwd()

capstone_dir = find_capstone_dir(base_dir)

os.makedirs(os.path.join(capstone_dir, 'src', 'output'), exist_ok=True)
os.makedirs(os.path.join(capstone_dir, 'src', 'input'), exist_ok=True)
os.makedirs(os.path.join(capstone_dir, 'src', 'models'), exist_ok=True)

input_dir = os.path.join(capstone_dir, 'src', 'input')
output_dir = os.path.join(capstone_dir, 'src', 'output')
models_dir = os.path.join(capstone_dir, 'src', 'models')


def pad_to_divisible_by_16(image: Image.Image) -> Image.Image:
    w, h = image.size
    pad_w = (16 - (w % 16)) % 16
    pad_h = (16 - (h % 16)) % 16
    padding = (0, 0, pad_w, pad_h)
    return transforms.functional.pad(image, padding)

class GenerateImageDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable[[Image.Image], Image.Image]] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    image_path = os.path.join(subdir, file)
                    self.image_paths.append(image_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_path

transform = transforms.Compose([
    transforms.Lambda(pad_to_divisible_by_16),
    transforms.ToTensor()
])

class UNet(nn.Module):
    def __init__(self, in_channels: int, num_masks: int) -> None:
        super(UNet, self).__init__()
        self.num_masks = num_masks
        
        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_masks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output_logit = self.final_conv(dec1)
        return output_logit

def load_model(model_path, num_masks, device):
    model = UNet(in_channels=3, num_masks=num_masks).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def run_inference(input_path, model_path, output_path, num_masks, device):
    dataset = GenerateImageDataset(input_path, transform=transforms.Compose([transforms.Lambda(pad_to_divisible_by_16), transforms.ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = load_model(model_path, num_masks, device)
    model_name = os.path.basename(model_path).replace('.pth', '')  

    with torch.no_grad():
        for images, image_paths in tqdm(dataloader, desc=f"Processing Model: {model_name}", bar_format="{l_bar}{bar} [ time left: {remaining} ]"):
            images = images.to(device)
            outputs = model(images)

            if num_masks == 1:
                predicted_mask = torch.sigmoid(outputs)
                predicted_mask = (predicted_mask > 0.5).float()
            else:
                predicted_mask = torch.softmax(outputs, dim=1)
                predicted_mask = torch.argmax(predicted_mask, dim=1)

            for i in range(predicted_mask.shape[0]):
                mask = predicted_mask[i].cpu().numpy()

                if mask.ndim == 2:
                    mask = mask[np.newaxis, ...]
                elif mask.ndim == 3:
                    mask = mask[0]

                mask = (mask * 255).astype('uint8')  
                image_path = image_paths[i]

                base_name, ext = os.path.splitext(os.path.basename(image_path))
                file_name = f"{base_name}_{model_name}.png"  
                save_path = os.path.join(output_path, file_name)

                Image.fromarray(mask).save(save_path)

description = """
Welcome to the Image Processing Program!
This program uses a UNet model to perform image segmentation on medical images.
You can load images from a specified directory, process them using trained models,
and save the output masks to an output directory. 

Enjoy the seamless experience! 
"""


microscope_art = """
          __
          ||
         ====
         |  |__
         |  |-.\\
         |__|  \\\\
          ||   ||
        ======__|
       ________||__
      /____________\\
      
"""


print(description)
print(microscope_art)

if __name__ == "__main__":
    while True:
        print("""   
            To easily use the program, paste your images into the input folder using your explorer. 
            Use the default settings to generate three masks - cancerous, inflammation, and hard to classify.
            You can also use an advanced configurator.""")

        use_defaults = input(""""
                             Would you like to use the default [def] or advanced [adv] settings? (def/adv): """)

        if use_defaults.lower() != 'def':
          
            change_input = input(f"Do you want to change the input directory? (yes/no) [default: {input_dir}]: ")
            if change_input.lower() == 'yes':
                input_dir = input("Enter new input_dir: ")
            
            print("Files in the input folder:")
            input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            
            if not input_files:
                print(f"Error: No images found in the input directory: {input_dir}")
                continue  

            for idx, file in enumerate(input_files, start=1):
                print(f"[{idx}] {file}")

            all_files = input("Would you like to use all files? (yes/no)): ")
            if all_files.lower() == 'no':
                selected_files = input("Enter the numbers of the images to process, separated by commas: ")
                selected_indices = [int(i.strip()) - 1 for i in selected_files.split(",")]
                input_files = [input_files[i] for i in selected_indices]

           
            change_output = input(f"Do you want to change the output directory? (yes/no)) [default: {output_dir}]: ")
            if change_output.lower() == 'yes':
                output_dir = input("Enter new output_dir: ")

            
            change_models = input(f"Do you want to change the models directory? (yes/no)) [default: {models_dir}]: ")
            if change_models.lower() == 'yes':
                models_dir = input("Enter new models_dir: ")

          
            print("Available models:")
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
            for idx, model in enumerate(model_files, start=1):
                print(f"[{idx}] {model}")

            selected_models = input("Select models to use, separated by commas: ")
            selected_indices = [int(i.strip()) - 1 for i in selected_models.split(",")]
            selected_models = [model_files[i] for i in selected_indices]

        else:
            
            selected_models = [
                ("cancerous.pth"),
                ("inflammatory.pth"),
                ("hard_to_classify.pth"),
            ]
            
            input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
                        
            if not input_files:
                print(f"Error: No images found in the input directory: {input_dir}")
                continue  
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if use_defaults.lower() == 'yes':
            
            for model_name in selected_models:
                model_path = os.path.join(models_dir, model_name)
                for input_file in input_files:
                    
                    run_inference(input_dir, model_path, output_dir, num_masks=1, device=device)
        else:
            
            for model_name in selected_models:
                model_path = os.path.join(models_dir, model_name)
                for input_file in input_files:                                      
                    run_inference(input_dir, model_path, output_dir, num_masks=1, device=device)

        print("All pictures processed.")
        continue_prompt = input("Would you like to do something else? (yes/no)): ")
        if continue_prompt.lower() != 'yes':
            print("Closing the program.")
            break