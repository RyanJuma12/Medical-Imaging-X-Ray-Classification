# %%
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# Create folders used to save model results
PLOTS_DIR = Path("../plots")
CHECKPOINT_DIR = Path("../checkpoint")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# Use the GPU when CUDA is available
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## Load Training, Validation, and Test Data
#
# The dataset has already been divided into training,
# validation, and testing folders.
#
# These datasets are loaded and prepared for model
# training and evaluation.

# %%
OUTPUT_DIR = Path("../img/output_test")

img_data = {}

for split_folder in sorted(OUTPUT_DIR.iterdir()):
    if not split_folder.is_dir():
        continue

    img_data[split_folder.name] = {}

    for class_folder in sorted(split_folder.iterdir()):
        if not class_folder.is_dir():
            continue

        image_count = sum(
            1
            for file in class_folder.iterdir()
            if file.is_file()
        )

        img_data[split_folder.name][class_folder.name] = image_count


df = pd.DataFrame.from_dict(
    img_data,
    orient="index"
)

print(df)

ax = df.T.plot(
    kind="bar",
    figsize=(8, 5)
)

ax.set_ylabel("Number of Images")
ax.set_xlabel("Chest X-Ray Class")
ax.set_title("Chest X-Ray Dataset Distribution")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "dataset_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# %% [markdown]
# ## Image Preprocessing
#
# Before training, every image is resized and normalized
# using the ImageNet preprocessing pipeline required for
# pretrained ResNet50 and VGG16 models.

# %%
im_size = 256

train_transform = transforms.Compose([
    transforms.Resize(300),

    transforms.RandomRotation(7),

    transforms.RandomHorizontalFlip(p=0.3),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.04, 0.04),
        scale=(0.95, 1.05)
    ),

    transforms.CenterCrop(im_size),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


evaluation_transform = transforms.Compose([
    transforms.Resize(300),

    transforms.CenterCrop(im_size),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_data = datasets.ImageFolder(
    OUTPUT_DIR / "train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    OUTPUT_DIR / "val",
    transform=evaluation_transform
)

test_data = datasets.ImageFolder(
    OUTPUT_DIR / "test",
    transform=evaluation_transform
)


BATCH_SIZE = 20

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


class_mapping = train_data.class_to_idx

print("Class mapping:", class_mapping)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))


# %% [markdown]
# ## Model Training
#
# Two pretrained convolutional neural networks are used
# for image classification:
#
# • ResNet50
# • VGG16
#
# The final classification layer is replaced so each model
# predicts one of the three chest X-ray classes:
#
# • CHF
# • Normal
# • Pneumonia

# %%
def validation(model, criterion, val_loader):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            total_loss += loss.item() * images.size(0)

            predictions = output.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_images += labels.size(0)

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images

    return average_loss, accuracy

# %%
def train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    epochs=30,
    model_name="model",
    scheduler=None,
    patience=5
):
    training_loss_history = []
    validation_loss_history = []
    training_accuracy_history = []
    validation_accuracy_history = []

    best_validation_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = 0

    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    for epoch in range(epochs):
        # -----------------------------
        # Training phase
        # -----------------------------
        model.train()

        total_training_loss = 0.0
        total_training_correct = 0
        total_training_images = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            predictions = output.argmax(dim=1)
            batch_size = labels.size(0)

            total_training_loss += (
                loss.item() * batch_size
            )

            total_training_correct += (
                predictions == labels
            ).sum().item()

            total_training_images += batch_size

        training_loss = (
            total_training_loss
            / total_training_images
        )

        training_accuracy = (
            total_training_correct
            / total_training_images
        )

        # -----------------------------
        # Validation phase
        # -----------------------------
        validation_loss, validation_accuracy = validation(
            model,
            criterion,
            val_loader
        )

        # Scheduler now monitors validation loss
        if scheduler is not None:
            scheduler.step(validation_loss)

        training_loss_history.append(training_loss)
        validation_loss_history.append(validation_loss)
        training_accuracy_history.append(training_accuracy)
        validation_accuracy_history.append(
            validation_accuracy
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch: {epoch + 1}/{epochs}.. "
            f"Training Loss: {training_loss:.3f}.. "
            f"Training Accuracy: "
            f"{training_accuracy:.3f}.. "
            f"Validation Loss: "
            f"{validation_loss:.3f}.. "
            f"Validation Accuracy: "
            f"{validation_accuracy:.3f}.. "
            f"Learning Rate: "
            f"{current_learning_rate:.6f}"
        )

        # Save model with lowest validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_accuracy = validation_accuracy
            best_epoch = epoch + 1

            best_model_weights = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print(
                f"New best {model_name} model at "
                f"epoch {best_epoch}"
            )

        else:
            epochs_without_improvement += 1

        # Early stopping disabled for this experiment.
        # The model will always complete every epoch.
        pass

    model.load_state_dict(best_model_weights)

    completed_epochs = len(training_loss_history)

    print(
        f"\nBest {model_name} model restored from "
        f"epoch {best_epoch}."
    )

    print(
        f"Validation accuracy at best epoch: "
        f"{best_accuracy:.3f}"
    )

    # -----------------------------
    # Loss graph
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, completed_epochs + 1),
        training_loss_history,
        label="Training Loss"
    )

    plt.plot(
        range(1, completed_epochs + 1),
        validation_loss_history,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"{model_name} Training and Validation Loss"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / f"{model_name.lower()}_training_loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # -----------------------------
    # Accuracy graph
    # -----------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, completed_epochs + 1),
        np.array(training_accuracy_history) * 100,
        label="Training Accuracy"
    )

    plt.plot(
        range(1, completed_epochs + 1),
        np.array(validation_accuracy_history) * 100,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(
        f"{model_name} Training and Validation Accuracy"
    )
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR
        / f"{model_name.lower()}_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return model

# %%
def save_checkpoint(
    model,
    class_mapping,
    arch,
    image_size=224
):
    """
    Save the trained model checkpoint.

    The checkpoint stores:
    - Model architecture
    - Class labels
    - Number of classes
    - Input image size
    - Learned model weights
    """

    checkpoint = {
        "arch": arch,
        "class_to_idx": class_mapping,
        "num_classes": len(class_mapping),
        "image_size": image_size,
        "model_state_dict": model.state_dict()
    }

    timestamp = datetime.now().strftime("%Y%m%d")

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{timestamp}_{arch}.pth"
    )

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print(
        f"Checkpoint saved to: "
        f"{checkpoint_path}"
    )

    return checkpoint_path

# %% [markdown]
# ## ResNet50 Model
#
# ResNet50 is a pretrained convolutional neural network that was
# originally trained on the ImageNet dataset.
#
# Rather than training an entirely new network, transfer learning is
# used by freezing the pretrained feature extraction layers and
# replacing the final classification layer.
#
# The modified network predicts one of three chest X-ray classes:
#
# - CHF
# - Normal
# - Pneumonia

# %%
resnet_model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

print(resnet_model)

# %% [markdown]
# ### Freeze and Fine-Tune ResNet50 Layers
#
# Most pretrained ResNet50 layers are frozen to preserve the
# features learned from ImageNet.
#
# The final two residual blocks, layer3 and layer4, are unfrozen
# so they can adapt to chest X-ray features.
#
# The final classification layer will also be replaced and trained
# to predict CHF, Normal, or Pneumonia.

# %%
# Freeze all pretrained layers first
for parameter in resnet_model.parameters():
    parameter.requires_grad = False

# Fine-tune the last two residual blocks
for parameter in resnet_model.layer3.parameters():
    parameter.requires_grad = True

for parameter in resnet_model.layer4.parameters():
    parameter.requires_grad = True

# %% [markdown]
# ### Replace the Classification Layer
#
# The original ImageNet classifier is replaced with a new classifier
# that predicts the three chest X-ray classes.
#
# A LogSoftmax output layer is used because the model is trained using
# Negative Log Likelihood Loss (NLLLoss).

# %%
print("Original final layer:")
print(resnet_model.fc)

num_classes = len(class_mapping)
num_features = resnet_model.fc.in_features

resnet_model.fc = nn.Sequential(
    OrderedDict([
        (
            "dropout",
            nn.Dropout(p=0.4)
        ),
        (
            "fc",
            nn.Linear(
                num_features,
                num_classes
            )
        ),
        (
            "output",
            nn.LogSoftmax(dim=1)
        )
    ])
)

resnet_model = resnet_model.to(device)

print(
    "ResNet50 device:",
    next(resnet_model.parameters()).device
)

print("\nModified final layer:")
print(resnet_model.fc)


# %% [markdown]
# ### Loss Function and Optimizer
#
# The model is trained using:
#
# - Negative Log Likelihood Loss (NLLLoss)
# - Adam Optimizer
#
# These are used to update the classifier weights while keeping the
# pretrained feature extraction layers frozen.

# %%
x = np.arange(0.01, 1.0, 0.001)
y = -np.log(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, y)

plt.ylabel("-log(x)")
plt.xlabel("x")
plt.title("Negative Log-Likelihood Range")

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "negative_log_likelihood.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# %% [markdown]
# The Adam optimizer is used to minimize the classification loss during
# training by updating the weights of the custom classifier.

# %%
resnet_criterion = nn.NLLLoss()

resnet_optimizer = optim.AdamW([
    {
        "params": resnet_model.layer3.parameters(),
        "lr": 0.00001
    },
    {
        "params": resnet_model.layer4.parameters(),
        "lr": 0.00003
    },
    {
        "params": resnet_model.fc.parameters(),
        "lr": 0.0001
    }
],
    weight_decay=0.0001
)

resnet_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    resnet_optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=0.000001
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in resnet_model.parameters()
    if parameter.requires_grad
)

print(
    f"Trainable ResNet50 parameters: "
    f"{trainable_parameters:,}"
)

# %% [markdown]
# ### Train the ResNet50 Model
#
# The customized ResNet50 model is trained using the augmented
# training dataset and evaluated after each epoch using the
# validation dataset.
# %%
resnet_model = train_model(
    model=resnet_model,
    optimizer=resnet_optimizer,
    criterion=resnet_criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=40,
    model_name="ResNet50",
    scheduler=resnet_scheduler,
    patience=7
)

# %% [markdown]
# ### Save the Trained Model
#
# Save the trained ResNet50 model so it can be used later for
# inference and Grad-CAM visualization.

# %%
#resnet_checkpoint_path = save_checkpoint(
#    model=resnet_model,
#    class_mapping=class_mapping,
#    arch="resnet50",
#    image_size=im_size
#)
# %%
resnet_model.eval()

resnet_predictions = []
resnet_true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = resnet_model(images)
        predictions = outputs.argmax(dim=1)

        resnet_predictions.extend(
            predictions.cpu().tolist()
        )

        resnet_true_labels.extend(
            labels.cpu().tolist()
        )

resnet_correct_predictions = sum(
    predicted == actual
    for predicted, actual in zip(
        resnet_predictions,
        resnet_true_labels
    )
)

resnet_test_accuracy = (
    resnet_correct_predictions
    / len(resnet_true_labels)
)

print(
    f"ResNet50 Test Accuracy: "
    f"{resnet_test_accuracy * 100:.1f}%"
)

# Save improved ResNet checkpoint
improved_resnet_checkpoint_path = (
    CHECKPOINT_DIR
    / "resnet50_layer3_layer4.pth"
)

improved_resnet_checkpoint = {
    "arch": "resnet50",
    "class_to_idx": class_mapping,
    "num_classes": len(class_mapping),
    "image_size": im_size,
    "test_accuracy": resnet_test_accuracy,
    "model_state_dict":
        resnet_model.state_dict()
}

torch.save(
    improved_resnet_checkpoint,
    improved_resnet_checkpoint_path
)

print(
    f"Improved ResNet50 checkpoint saved to: "
    f"{improved_resnet_checkpoint_path}"
)

resnet_report = classification_report(
    resnet_true_labels,
    resnet_predictions,
    target_names=test_data.classes,
    digits=3,
    zero_division=0
)

print(resnet_report)


resnet_report_path = (
    PLOTS_DIR
    / "resnet50_classification_report.txt"
)

with open(
    resnet_report_path,
    "w",
    encoding="utf-8"
) as report_file:
    report_file.write(resnet_report)

print(
    f"Classification report saved to: "
    f"{resnet_report_path}"
)


# %%
resnet_confusion_matrix = confusion_matrix(
    resnet_true_labels,
    resnet_predictions
)

print("ResNet50 confusion matrix:")
print(resnet_confusion_matrix)


fig, ax = plt.subplots(figsize=(7, 6))

matrix_image = ax.imshow(
    resnet_confusion_matrix
)

ax.set_title("ResNet50 Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

class_names = test_data.classes

ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)


threshold = resnet_confusion_matrix.max() / 2

for row in range(resnet_confusion_matrix.shape[0]):
    for column in range(
        resnet_confusion_matrix.shape[1]
    ):
        value = resnet_confusion_matrix[
            row,
            column
        ]

        text_color = (
            "white"
            if value > threshold
            else "black"
        )

        ax.text(
            column,
            row,
            value,
            ha="center",
            va="center",
            color=text_color
        )


fig.colorbar(
    matrix_image,
    ax=ax
)

plt.tight_layout()

resnet_cm_path = (
    PLOTS_DIR
    / "resnet50_confusion_matrix.png"
)

plt.savefig(
    resnet_cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(
    f"Confusion matrix saved to: "
    f"{resnet_cm_path}"
)

# %% [markdown]
# ### Freeze and Fine-Tune VGG16 Layers
#
# Most pretrained VGG16 layers are frozen to preserve the
# features learned from ImageNet.
#
# The final convolutional block is unfrozen so it can adapt
# to chest X-ray features.
#
# The final classification layer is replaced to predict:
#
# - CHF
# - Normal
# - Pneumonia
# %%
vgg_model = models.vgg16(
    weights=models.VGG16_Weights.DEFAULT
)

print(vgg_model)

# %%
# Freeze all pretrained VGG16 parameters
for parameter in vgg_model.parameters():
    parameter.requires_grad = False


# Fine-tune the final convolutional block
for parameter in vgg_model.features[24:].parameters():
    parameter.requires_grad = True

# %%
print("Original final layer:")
print(vgg_model.classifier[6])

num_classes = len(class_mapping)
num_features = vgg_model.classifier[6].in_features

vgg_model.classifier[6] = nn.Sequential(
    nn.Linear(
        num_features,
        num_classes
    ),
    nn.LogSoftmax(dim=1)
)

vgg_model = vgg_model.to(device)

print(
    "VGG16 device:",
    next(vgg_model.parameters()).device
)

print("\nModified final layer:")
print(vgg_model.classifier[6])

# %% [markdown]
# ### Train the VGG16 Model
#
# The customized VGG16 model is trained using the same loss function,
# optimizer, and training procedure as the ResNet50 model.
# %%
vgg_criterion = nn.NLLLoss()

vgg_optimizer = optim.Adam([
    {
        "params": vgg_model.features[24:].parameters(),
        "lr": 0.00001
    },
    {
        "params": vgg_model.classifier.parameters(),
        "lr": 0.0001
    }
])

vgg_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    vgg_optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=0.000001
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in vgg_model.parameters()
    if parameter.requires_grad
)

print(
    f"Trainable VGG16 parameters: "
    f"{trainable_parameters:,}"
)

# %%
vgg_model = train_model(
    model=vgg_model,
    optimizer=vgg_optimizer,
    criterion=vgg_criterion,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    model_name="VGG16",
    scheduler=vgg_scheduler,
    patience=5
)

# %% [markdown]
# ### Save the Trained Model
#
# Save the trained VGG16 model for later evaluation and comparison
# with the ResNet50 model.
# %%
vgg_checkpoint_path = save_checkpoint(
    model=vgg_model,
    class_mapping=class_mapping,
    arch="vgg16",
    image_size=im_size
)

# %% [markdown]
# ## Model Evaluation
#
# After training, the models are evaluated using the test dataset.
#
# Performance is summarized using a classification report containing:
#
# - Precision
# - Recall
# - F1-score
# - Overall accuracy
#
# These metrics provide a detailed assessment of each model's
# performance on unseen chest X-ray images.
# %%
vgg_model.eval()

vgg_predictions = []
vgg_true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = vgg_model(images)
        predictions = outputs.argmax(dim=1)

        vgg_predictions.extend(
            predictions.cpu().tolist()
        )

        vgg_true_labels.extend(
            labels.cpu().tolist()
        )

vgg_correct_predictions = sum(
    predicted == actual
    for predicted, actual in zip(
        vgg_predictions,
        vgg_true_labels
    )
)

vgg_test_accuracy = (
    vgg_correct_predictions
    / len(vgg_true_labels)
)

print(
    f"VGG16 Test Accuracy: "
    f"{vgg_test_accuracy * 100:.1f}%"
)

vgg_report = classification_report(
    vgg_true_labels,
    vgg_predictions,
    target_names=test_data.classes,
    digits=3,
    zero_division=0
)

print(vgg_report)


vgg_report_path = (
    PLOTS_DIR
    / "vgg16_classification_report.txt"
)

with open(
    vgg_report_path,
    "w",
    encoding="utf-8"
) as report_file:
    report_file.write(vgg_report)

print(
    f"Classification report saved to: "
    f"{vgg_report_path}"
)

# %%
vgg_confusion_matrix = confusion_matrix(
    vgg_true_labels,
    vgg_predictions
)

print("VGG16 confusion matrix:")
print(vgg_confusion_matrix)


fig, ax = plt.subplots(figsize=(7, 6))

matrix_image = ax.imshow(
    vgg_confusion_matrix
)

ax.set_title("VGG16 Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

class_names = test_data.classes

ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))

ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)


threshold = vgg_confusion_matrix.max() / 2

for row in range(vgg_confusion_matrix.shape[0]):
    for column in range(
        vgg_confusion_matrix.shape[1]
    ):
        value = vgg_confusion_matrix[
            row,
            column
        ]

        text_color = (
            "white"
            if value > threshold
            else "black"
        )

        ax.text(
            column,
            row,
            value,
            ha="center",
            va="center",
            color=text_color
        )


fig.colorbar(
    matrix_image,
    ax=ax
)

plt.tight_layout()

vgg_cm_path = (
    PLOTS_DIR
    / "vgg16_confusion_matrix.png"
)

plt.savefig(
    vgg_cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(
    f"Confusion matrix saved to: "
    f"{vgg_cm_path}"
)

#%%
model_names = [
    "ResNet50",
    "VGG16"
]

model_accuracies = [
    resnet_test_accuracy * 100,
    vgg_test_accuracy * 100
]

plt.figure(figsize=(7, 5))

bars = plt.bar(
    model_names,
    model_accuracies
)

plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.title("ResNet50 vs. VGG16 Test Accuracy")

plt.ylim(
    0,
    max(model_accuracies) + 15
)

for bar, accuracy in zip(
    bars,
    model_accuracies
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.1f}%",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold"
    )

plt.tight_layout()

accuracy_comparison_path = (
    PLOTS_DIR
    / "model_accuracy_comparison.png"
)

plt.savefig(
    accuracy_comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(
    f"Accuracy comparison saved to: "
    f"{accuracy_comparison_path}"
)

# %% [markdown]
# ## Possible Model Improvements
#
# Possible future improvements include:
#
# - Use class-weighted loss if one disease class has lower recall
# - Experiment with a learning-rate scheduler
# - Test a larger image size such as 320 × 320
# - Use early stopping to reduce overfitting
# - Compare Grad-CAM regions with expert eye-gaze heatmaps
# - Increase the dataset size with additional real chest X-rays
#
# Augmentation should be applied only to the training dataset.
# Validation and test images should not be randomly augmented.

# %% [markdown]
# ## Load Saved Models
#
# The following function reloads a saved model checkpoint for
# inference or future visualization using Grad-CAM.

# %%
def load_checkpoint(filepath):
    """
    Load a saved ResNet50 or VGG16 checkpoint.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved checkpoint.

    Returns
    -------
    model
        Rebuilt model loaded with the saved weights.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        print(f"No checkpoint found at: {filepath}")
        return None

    checkpoint = torch.load(
        filepath,
        map_location=device
    )

    architecture = checkpoint["arch"]
    class_to_idx = checkpoint["class_to_idx"]

    num_classes = checkpoint.get(
        "num_classes",
        len(class_to_idx)
    )

    if architecture == "resnet50":
        loaded_model = models.resnet50(
            weights=None
        )

        num_features = loaded_model.fc.in_features

        loaded_model.fc = nn.Sequential(
            OrderedDict([
                (
                    "dropout",
                    nn.Dropout(p=0.4)
                ),
                (
                    "fc",
                    nn.Linear(
                        num_features,
                        num_classes
                    )
                ),
                (
                    "output",
                    nn.LogSoftmax(dim=1)
                )
            ])
        )

    elif architecture == "vgg16":
        loaded_model = models.vgg16(
            weights=None
        )

        num_features = (
            loaded_model.classifier[6].in_features
        )

        loaded_model.classifier[6] = nn.Sequential(
            nn.Linear(
                num_features,
                num_classes
            ),
            nn.LogSoftmax(dim=1)
        )

    else:
        print(
            f"Architecture not recognized: "
            f"{architecture}"
        )
        return None

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.class_to_idx = class_to_idx
    loaded_model.image_size = checkpoint.get(
        "image_size",
        224
    )

    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    print(
        f"Loaded {architecture} checkpoint from: "
        f"{filepath}"
    )

    return loaded_model

# %% [markdown]
# ### VGG16 Results

# %%
model = load_checkpoint(vgg_checkpoint_path)
# %% [markdown]
# ## Test Loading a Saved Model
#
# Reload the saved VGG16 checkpoint to confirm that the
# checkpoint can be used later for inference or Grad-CAM.

# %%
loaded_vgg_model = load_checkpoint(
    vgg_checkpoint_path
)

if loaded_vgg_model is not None:
    print("VGG16 checkpoint loaded successfully.")
    print(
        "Loaded class mapping:",
        loaded_vgg_model.class_to_idx
    )
# %%