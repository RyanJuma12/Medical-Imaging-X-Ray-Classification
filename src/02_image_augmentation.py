# %% [markdown]
# # Medical Imaging AI: 02 Image Augmentation
#
# This script prepares the chest X-ray dataset for model training.
#
# The dataset is divided into training, validation, and testing sets.
# Data augmentation is then applied to the training images to reduce
# class imbalance and improve the model's ability to generalize.
#
# Dataset Classes:
# - CHF
# - Normal
# - Pneumonia
# %%
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import splitfolders

# Augment images
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# %%
DATA_DIR = "../img/Dataset/"
img_data = {}

for folder in os.listdir(DATA_DIR):
    img_path = DATA_DIR + folder + '/'
    img_data[folder] = [img_path + img for img in os.listdir(img_path)]

# %% [markdown]
# ## Dataset Distribution
#
# Display the number of images in each chest X-ray class before
# splitting and augmentation.

# %% [markdown]
# ## Chest X-ray Class Distribution
#
# Visualize the number of images available in each class before
# augmentation.

# %%
# Count the number of images in each chest X-ray class
xray_classes = {
    "CHF": len(img_data['CHF']),
    "Normal": len(img_data['Normal']),
    "Pneumonia": len(img_data['Pneumonia'])
}

# Plot the distribution of the dataset
plt.bar(xray_classes.keys(), xray_classes.values())
plt.ylabel('Number of Images')
plt.xlabel('Chest X-ray Classes')
plt.title('Distribution of Chest X-ray Images by Class')
plt.show()

# %% [markdown]
# ## Split Dataset
#
# Divide the dataset into training, validation, and testing sets.
#
#The dataset is divided using a 75% training, 12.5% validation,
#and 12.5% testing split.
#
# The remaining images are used for training and augmentation.

# %%
DATA_DIR = "../img/Dataset/"
OUTPUT_DIR = "../img/output_test/"

splitfolders.ratio(
    DATA_DIR,
    output=OUTPUT_DIR,
    seed=1337,
    ratio=(0.75, 0.125, 0.125)
)

# %% [markdown]
# ## Data Augmentation
#
# Data augmentation increases the diversity of the training dataset
# by generating modified versions of existing chest X-ray images.
#
# The following augmentations are applied:
#
# - Small random rotations
# - Horizontal flipping
# - Random grayscale conversion
# - Sharpness adjustment
# - Brightness and contrast variation
# - Gaussian blur
#
# These transformations help reduce overfitting and improve the
# model's ability to generalize to unseen images.
# 

# %%
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(7),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05)
    ),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

img = Image.open(img_data['CHF'][0])
fig, axarr = plt.subplots(1,4)
aug_test = [image_transforms(img).permute(1, 2, 0) for i in range(3)]
for ax, im in zip(axarr, [img]+aug_test):
    ax.imshow(im)
    ax.axis('off')
plt.show()


# %% [markdown]
# ## Balance the Training Dataset
#
# Generate augmented images for classes with fewer training samples
# until all classes contain approximately the same number of images.

# %%
train_data = {}

for folder in os.listdir(OUTPUT_DIR+'train/'):
    img_path = OUTPUT_DIR+'train/' + folder + '/'
    train_data[folder] = [img_path + img for img in os.listdir(img_path)]

max_num_img = max([len(v) for k,v in train_data.items()])
print(f"Number of images in largest class: {max_num_img}")

# %%
for folder, path_list in train_data.items():
    if len(path_list) < max_num_img:
        sets = (max_num_img-len(path_list)) // len(path_list)
        mod = (max_num_img-len(path_list)) % len(path_list)
        for i, path in enumerate(path_list):
            img = Image.open(path)
            img = img.convert('RGB')
            sets_iter = sets + 1 if i < mod else sets
            for k in range(sets_iter):
                save_image(image_transforms(img), f'{OUTPUT_DIR}/train/{folder}/augmented_{i}_{k}.jpg')

# %% [markdown]
# ## Final  Dataset Slpit
#
# Display the number of images in each training class after
# augmentation to verify that the dataset is balanced.

# %%
split_counts = {}

for split in ["train", "val", "test"]:
    split_counts[split] = {}

    for class_name in ["CHF", "Normal", "Pneumonia"]:
        folder_path = os.path.join(
            OUTPUT_DIR,
            split,
            class_name
        )

        split_counts[split][class_name] = len(
            os.listdir(folder_path)
        )

df_counts = pd.DataFrame(split_counts)

print(df_counts)

ax = df_counts.plot(
    kind="bar",
    figsize=(8,5)
)

ax.set_ylabel("Number of Images")
ax.set_xlabel("Chest X-ray Class")
ax.set_title("Final Train / Validation / Test Distribution")

plt.tight_layout()
plt.show()


# %%
