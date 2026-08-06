# %% [markdown]
# # Medical Imaging AI: 01 Image Cleaning
#
# This script performs the initial cleaning of the chest X-ray dataset
# used for training deep learning models.
#
# The dataset contains three classes:
#
# - CHF
# - Normal
# - Pneumonia
#
# Before training, the dataset is checked for corrupted images,
# duplicate images, and images that have been manually identified
# as poor-quality samples.
#
# Cleaning the dataset helps improve training quality and reduces
# bias introduced by duplicate or invalid images.

# %%
import os
import json
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load images
from PIL import Image
from skimage import img_as_float

# %%
DATA_DIR = "../img/Dataset/"
img_data = {}

for folder in os.listdir(DATA_DIR):
    img_path = DATA_DIR + folder + '/'
    img_data[folder] = [img_path + img for img in os.listdir(img_path)]


# %% [markdown]
# ## Verify Image Files
#
# Before training, every file is opened to verify that it is a valid
# image. Corrupted or unreadable files can cause training to fail,
# so this step reports any invalid images found in the dataset.

# %%
print("Checking dataset...")

for folder, img_list in img_data.items():

    counter = 0

    for path in img_list:

        try:

            Image.open(path)

        except IOError:

            counter += 1

            print(path)

    print(f"{folder}: {counter} invalid images")

# %% [markdown]
# ## Load Manual Removal List
#
# The file remove_img_path.json contains images that were manually
# reviewed and selected for removal because they were poor quality,
# incorrectly labeled, or otherwise unsuitable for training.

# %%
with open("../img/remove_img_path.json", "r") as f:
    remove_img_path = json.load(f)

# %% [markdown]
# ## Remove Images
#
# Remove all images listed in remove_img_path.json from the dataset.

# %%
for folder, path_list in remove_img_path.items():

    for path in path_list:

        if os.path.isfile(path):

            os.remove(path)

# %%
