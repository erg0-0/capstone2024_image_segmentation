![logo_UMP](https://gitlab.com/limeryk/capstone_group5/-/blob/main/documents/logo_UMP.png?ref_type=heads)
![logo_Roche](https://gitlab.com/limeryk/capstone_group5/-/raw/main/documents/logo_Roche.png?ref_type=heads)


**Medical University in Poznan - Data Science in Medicine**

# Project Capstone Group 5

Project has been developed between 11th August 2024 and 24th September 2024 as a final project for the 3-term post-diploma studies **Data Science in Medicine** at the Medical University in Poznan, Poland.
Project brief can be found [here (PL)](https://gitlab.com/limeryk/capstone_group5/-/blob/main/documents/capstone_wymagania.pdf?ref_type=heads) and requirements are [here (PL)](https://gitlab.com/limeryk/capstone_group5/-/blob/main/documents/capstone_wymagania.pdf?ref_type=heads).

 **Project Mentor:**  
Mateusz Bednarski 

 **Project Members:**
  - Liliana Gmerek
  - MD Marta Krysik
  - MD Jakub Miąskowski

**Data Problem Type:** \
Computer Vision * Deep Learning * Image Segmentation

**Acknowledgments:**
We would like to express our gratitude for providing access to the dataset and for introducing us to Dr. inż. Adam Kozak. \
We would like to extend our sincere thanks to our project mentor, Mateusz Bednarski, for his invaluable help and support.



# General problem description

## Introduction
**Lung cancer** is one of the most common and deadly malignancies worldwide. 

**Non-Small Cell Lung Cancer (NSCLC)**, which constitutes about 85% of all lung cancer cases, often presents at an advanced stage due to the absence of early symptoms. Early detection and precise diagnosis are crucial for improving treatment outcomes and survival rates.

Deep learning (DL) offers a promising opportunity to aid pathologists in cancer diagnosis. By leveraging computer vision techniques, DL models have the potential to assist in the analysis of histopathological images, identifying subtle patterns and abnormalities in tissue samples with greater precision and speed. This technology could significantly reduce diagnostic time, enhance accuracy, and support pathologists in providing earlier, more reliable diagnoses, ultimately improving treatment decisions and patient outcomes.


### DATASET SOURCE

The models developed in this project are based on a dataset prepared by a team of pathologists working closely with **Roche company**. 

The dataset consists of selected regions from microscopic scans that include cancerous cells, with **expert-annotated masks** representing **four distinct classes of tissue**. 

These classes include:
- **Cancer tissue** – Regions containing malignant cells.
- **Inflammatory tissue** – Areas of inflammation, which must be distinguished from cancerous regions.
- **Hard to classify** – Regions with ambiguous tissue characteristics.
- **Stroma** – Connective tissue that supports the cancerous areas. _Due to the limited number of examples of stroma in the dataset, this class was excluded from the final segmentation process. _


### PROJECT GOAL

**The model focuses on the three classes of tissues: cancerous, inflammatory and hard to classify,** improving the ability to SEGMENT and CLASSIFY liquid biopsy samples efficiently. 

This project leverages **convolutional neural networks (CNNs), specifically U-Net architecture ** to automate the segmentation process, aiding pathologists in making faster and more accurate diagnoses for NSCLC patients.


In this README, we provide an overview of the dataset, model architecture, and the training process, along with instructions for reproducing the results and evaluating the model.

# Getting started - configuration

## Connect to the git repository

1. Open a terminal in a desired location.
2. `git clone https://gitlab.com/limeryk/capstone_group5.git`

## Download data
1. Download data from [here](https://www.kaggle.com/datasets/kozaka2/histopathology-segmentation-markers). Access is required.
2. Paste the folders into the git repository project structure into folder **src/data** (compare project structure below).


## Project structure

Expected project structure. Folder data must be created manually, other folders are created automatically.

**capstone_group5** (git repository)

├── documents  
├── src  
│   ├── data  
│   ├── models  
│   ├── notebooks  
│   ├── reports  
│   ├── unit_tests  
│   └── utils  

## Download models

Training of the models may take many hours. In order to use pre-trained models, download them and place in the folder **src/models**. 
Download from **[here](https://drive.google.com/drive/folders/1f52lCgw3u9wAVr8B5jR3gtHXyivFWOCC?usp=sharing)**.

## Environments

Used libraries are stored in the requirements.txt in the main catalogue. 
For setting environments there've been used both conda and pip.
We have used the newest version - Python 3.12.5.

# How to run the model?

## Method 1 - Command Line Interface

In the main catalogue, the command line tool **cli.py** allows interaction with the model. 

- **Input: one or more images in format of ".jpg", ".jpeg", ".png", ".bmp", or ".tiff"
- **Output: black& white image(s) with segmented masks


1. In the terminal enter the main catalogue.
2. Open cli.py from the main catalogue and read the instructions.
3. Paste your images in input folder, located in src.
3. Type 'def' and press enter for default settings. The default segmentation models will load 3 classes: cancerous, inflammatory, hard to classify.
4. After the segmentation is finished, the generated masks are to be found in src/output folder. 
5. You can use advanced configuration for setting your own input and output directories. Choice of other models is also possible.

## Method 2 - REVIEWING THE CODE DIRECTLY

The code could be also accessed directly. Accessing it by using the below methods allows experimenting. Besides the training, the code generates the following outputs for further analysis:
1. sample images vs true mask
2. sample images vs true mask vs predicted mask
3. charts of loss function over epochs 
4. charts of dice coefficient over epochs
5. excel report with all the model parameters (stored automatically in src/reports). In order to use the functionality ensure that **experiment_id* is unique per each of your experiment. Then a new line in the report will be added with all applied parameters and metrics.

## Method 2-A - Reviewing the code in the histopathology.py.


The code could be also accessed directly over script.
It is possible to train own models with own parameters.

### Overview of the configuration of the model.

Configurable parameters are collected in a dedicated class CFG (Configuration). 
The CFG class contains all the hyperparameters and settings required to train, validate, and test the deep learning model. Below is a detailed description of each parameter:

| Parameter      | Description                                                                  | Example  |
|----------------|------------------------------------------------------------------------------|----------|
| experiment_id  | A unique identifier for the current experiment. Helps to distinguish different experiments in logs and reports | 'JM50'   |
| model_name         | Specifies the type of model being used for training. In this case, it is set to 'UNet', which is a widely used model for image segmentation tasks.                                                                       | 'UNet'   |
| train_continuation         | Boolean flag that indicates whether to continue training from a previously saved model. If True, it loads the pretrained model specified in pretrained_model_path.                                                                       | False   |
|pretrained_model_path | Path to the pretrained model. If train_continuation is True, the model weights from this file will be loaded to continue training. | 'type_your_model_directory_here'|
|early_stopping | Enables or disables early stopping. Early stopping is a technique where training stops if the model's performance on the validation set stops improving for a number of epochs. |False |
|es_patience| The number of epochs to wait before stopping the training when early stopping is enabled. If the validation performance does not improve after es_patience epochs, training will stop. | 3 |
|es_delta| The minimum change in validation performance to be considered as an improvement when early stopping is enabled. Smaller values make the training more sensitive to minor improvements. | 0.002 |
| train_bs | The batch size for the training set. This is the number of samples processed before the model updates its internal parameters. | 4 |
| valid_bs | The batch size for the validation set. |  4 |
| crop_size| Defines the size of image crops used for training. Each image will be cropped to this size | (512, 512) (width, height in pixels) |
| num_crops | The number of random crops to take from each image during training. Helps in augmenting the dataset for training. | 8 |
| epochs | The number of epochs (iterations over the entire dataset) for training. | 100 |
|lr | Learning rate for the optimizer. Controls how much the model’s parameters are adjusted during training. Smaller values lead to slower learning. |0.0001 |
| data_train_test_split | Ratio for splitting the data into training and testing sets. The value represents the proportion of the data used for training | 0.9 (90% training, 10% testing) | 
|data_train_val_split | Ratio for splitting the training data into training and validation sets. The value represents the proportion of training data used for training (the rest goes to validation) | 0.8 (80% training, 20% validation from the training dataset) |
|device |Specifies the device used for training (CPU or GPU). The model automatically selects CUDA (GPU) if available, otherwise defaults to CPU. | torch.device("cuda:0" if torch.cuda.is_available() else "cpu")|




#### Mask Training Flags

The following flags allow you to configure which types of segmentation masks should be considered during model training. Based on the combination of flags set to True, you can train the model for single-class or multi-class segmentation.

1. If only one flag is set to True, the model will be trained for single-class segmentation. 

2. If multiple flags are set to True, the model will be trained for multi-class segmentation. 

The results obtained from the single-class models are superior, and as such, they are presented in the analysis.

| Parameter | Description | Example |
| ------ | ------ | ------ | 
|    mask_train    |  A flag indicating whether to train the model to segment all classes combined. Set this flag to True if the sum of all masks is part of the model's objective.      | True / False |
|    cancerous_train     |    A flag indicating whether to train the model to segment cancerous tissue. Set this flag to True if cancerous tissue segmentation is part of the model's objective.    |  True / False |
| hard_to_classify_train | A flag indicating whether to train the model to segment tissue that is hard to classify. Set to True if your dataset contains this type of tissue and it's important for the segmentation task.| True / False |
| inflammatory_train | A flag indicating whether to train the model to segment inflammatory tissue. Useful if the model is intended to differentiate between cancerous and inflammatory regions. | True / False |
|stroma_train | A flag indicating whether to train the model to segment stroma tissue. Due to insufficient examples of stroma in the dataset,  we didnt't train a capable model. |  True / False |


## Method 2-B - Running histopathology_starter.ipynb in Kaggle

The notebook is set up to be used in Kaggle directly.
Choose the following notebook: **src/notebooks/histopathology_starter.ipynb 
When working with the file in Kaggle, it is need to be ensured that also the project catalogue structure with functions is copied. 

1. Find the main catalogue (capstone_group5).
- copy it without the folders "src/data", "src/input" and "src/output" where the page dataset of images is located. 
- Keep everything else in the project, including code, notebooks, and other necessary files.
- Compress the entire project directory (capstone_group5) into a .zip file.
2. Identify the Existing Kaggle Dataset: the Kaggle dataset ID is **histopathology-segmentation-markers** . 
3: Upload Your Project Without the Data Files
- Go to the Kaggle website and log in to your account.
- Click on your profile icon (top right) and select "Your Datasets".
- On the new page, click "New Dataset".
- Select "Upload a dataset".
- Drag and drop the zipped file of your project (without the data) into the upload area.
- Add a description and title for the project dataset (optional).
- Click "Create" to upload the project.
4.Create a Kaggle Notebook to Link the Existing Dataset
- After uploading your project files, go to the Kaggle Notebooks section.
- Click "New Notebook".
- In the new notebook interface, click on "Add Data" on the right side of the page.
- In the search box, type the name of the existing dataset you want to link to.
- Select that dataset, and it will be added to your notebook. You can now access this dataset at /kaggle/input/dataset-name.
5: Test Your Project
Run the notebook in Kaggle to ensure that your project works with the Kaggle dataset. Make sure all references to the dataset files work correctly using the new Kaggle path.


## Using pre-trained models

Training the model can take very long time. In case you want to train model in the different machine and work on it later locally, here's description on how to enable it.
1. Saving model after training in a dedicated machine or environment.
After finishing the training, the models are automatically stored in the src/models. Size of these files may be considerable. Model's parameter values are  automatically stored in src/reports.
2. Reading the model for the first time by different user/different machine:
- paste the model to src/models
- in the hystopatology.py or hystopathology_starter.ipynb check the definition of class CFG
- change the following parameters:
  - train_continuation      = True
  - pretrained_model_path   = 'type_your_model_directory_path_here'
3.  Ensure that the following parameters are set to the same values that were used during the initial training to avoid shape mismatches in the model:
- train_bs (training batch size)
- valid_bs (validation batch size)
- crop_size (image crop size)
- num_crops (number of crops per image)
- mask_train (whether masks are used during training)
- cancerous_train (train to segment cancerous tissue)
- hard_to_classify_train (train to segment hard-to-classify tissue)
- inflammatory_train (train to segment inflammatory tissue)
- stroma_train (train to segment stroma tissue)
4. Set the remaining hyperparameters:
- lr (learning rate), 
- epochs (number of epochs), 
- early stopping, 
- es_patience
-  es_delta

as needed for the continuation of the training.

5. Parameter values can be obtained from the generated report in the original machine.



## Additional analysis

Some additional analysis needed for the project has been performed and stored in src/notebooks.