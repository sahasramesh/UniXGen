# UniXGen
Code for the CHIL 2024 paper:
["Vision-Language Generative Model for View-Specific Chest X-ray Generation"](https://arxiv.org/abs/2302.12172)

# Project Introduction 
The Github Repository attempts to reproduce the results of the paper ["Vision-Language Generative Model for View-Specific Chest X-ray Generation"](https://arxiv.org/abs/2302.12172). In this research paper they propose a new vision-language generative model called ViewXGen. ViewXGen can generate chest X-rays from specific angles, even if those images are not in the training set. It does so by using specialized tokens for each type of view, frontal, lateral, and oblique. 

However, due to python version discrepenicy issues between the code from the paper and our code in google colab, we were unable to reproduce the results. Nevertheless, the project covers essential steps like dataset loading, preprocessing, and model creating, based on paper.

# Potential Steps to Reproduce the Paper Using Python 3.11.1

## Step 1: Clone This Repository
~~~
!git clone https://github.com/sahasramesh/UniXGen.git
~~~

## Step 2: Install all the dependencies (except a few)
~~~
!pip install -r requirements.txt
~~~

## Step 3: Install Torch
~~~
!pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
~~~

## Step 4: Install Torch_xla to use with Google Colab TPU
~~~
!pip install torch_xla[tpu]==2.6.0 -f https://storage.googleapis.com/libtpu-releases/index.html
~~~

## Step 5 Install Model Weights

####  1. [Chest X-ray Tokenizer](https://drive.google.com/drive/folders/1Ia_GqRrmZ8g6md02TC5_nkrGn6eUwVaG?usp=sharing): Download VQGAN and place into the /mimiccxr_vqgan directory
####  2. [UniXGen](https://drive.google.com/file/d/1RBQEoYTBRKBh6L53QCE0OIXL0Da5knkY/view?usp=sharing): Download the model and place into the /ckpt directory

~~~
# Step 1: Install gdown
!pip install -q gdown

# Step 2: Create the required directories
!mkdir -p /ckpt

# Step 3: Download Chest X-ray Tokenizer (VQGAN)
!gdown --folder --output /mimiccxr_vqgan https://drive.google.com/drive/folders/1Ia_GqRrmZ8g6md02TC5_nkrGn6eUwVaG

# Step 4: Download UniXGen Model Weights
!gdown --id 1RBQEoYTBRKBh6L53QCE0OIXL0Da5knkY --output /ckpt
~~~

## Step 6: Install Dataset
1. You must be a credential user defined in [PhysioNet](https://physionet.org/settings/credentialing/) to access the data.
2. Download chest X-rays from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) and reports from [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)
3. We provide train, valid and test split sets in /metadata directory.

~~~
import os
import random
from google.colab import auth

# Step 1: Authenticate
auth.authenticate_user()

PROJECT_ID = "deeplearning-459114"
DATASET_DIR = "/content/data"

# Step 2: Create necessary directories
!mkdir -p "${DATASET_DIR}/mimic-cxr-jpg/p10"
!mkdir -p "${DATASET_DIR}/mimic-cxr/p10"

# Step 3: List all patient folders from mimic-cxr-jpg (assuming both datasets have the same patient IDs)
all_patient_folders = !gsutil -u $PROJECT_ID ls gs://mimic-cxr-jpg-2.0.0.physionet.org/files/p10/
all_patient_ids = [folder.strip().split('/')[-2] for folder in all_patient_folders if folder.strip()]

# Step 4: Randomly select 100 unique patient IDs
selected_patient_ids = random.sample(all_patient_ids, 100)

# Step 5: Copy those 100 patient folders from both datasets
for patient_id in selected_patient_ids:
    !gsutil -u $PROJECT_ID cp -r gs://mimic-cxr-jpg-2.0.0.physionet.org/files/p10/{patient_id} "${DATASET_DIR}/mimic-cxr-jpg/p10/"
    !gsutil -u $PROJECT_ID cp -r gs://mimic-cxr-2.0.0.physionet.org/files/p10/{patient_id} "${DATASET_DIR}/mimic-cxr/p10/"

# Step 6: Optionally check
!ls -l "${DATASET_DIR}/mimic-cxr-jpg/p10/" | wc -l
!ls -l "${DATASET_DIR}/mimic-cxr/p10/" | wc -l
~~~
## Train Models

~~~
python unified_main.py
~~~

## Test Models

First, run unified_run.py. \
The generated discrete code sequences are saved as files.
  
~~~
python unified_run.py
~~~

#### For decoding chest X-rays,
Run decode_cxr.py. \
The generated seqeucens for chest X-rays are decoded and saved in the '.jpeg' format.

~~~
python decode_cxr.py
~~~

#### For decoding radiology reports,
Run decode_report.py. \
Save the decoded outputs according to your preference.

~~~
python decode_report.py
~~~
