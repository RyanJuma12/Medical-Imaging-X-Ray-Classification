# %%
from pathlib import Path
from collections import OrderedDict

import gradio as gr
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


# %%
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = Path(
    "../checkpoint/20260727_resnet50.pth"
)

print("Device:", DEVICE)


# %%
def load_checkpoint(filepath):
    """
    Rebuild and load a saved ResNet50 or VGG16 model.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {filepath}"
        )

    checkpoint = torch.load(
        filepath,
        map_location=DEVICE
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
        raise ValueError(
            f"Unsupported architecture: {architecture}"
        )

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.class_to_idx = class_to_idx
    loaded_model.idx_to_class = {
        index: class_name
        for class_name, index
        in class_to_idx.items()
    }

    loaded_model = loaded_model.to(DEVICE)
    loaded_model.eval()

    print(
        f"Loaded {architecture} checkpoint from "
        f"{filepath}"
    )

    return loaded_model


# %%
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def prepare_image(image):
    """
    Convert an uploaded PIL image into a model tensor.
    """

    image = image.convert("RGB")
    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(DEVICE)


# %%
def predict_image(image):
    """
    Predict the class and confidence values for an X-ray.
    """

    if image is None:
        return None, {
            "No image uploaded": 0.0
        }

    image_tensor = prepare_image(image)

    with torch.no_grad():
        log_probabilities = model(image_tensor)
        probabilities = torch.exp(
            log_probabilities
        )[0]

    confidence_scores = {
        model.idx_to_class[index]:
        float(probabilities[index].item())
        for index in range(
            len(model.idx_to_class)
        )
    }

    return image, confidence_scores


# %%
model = load_checkpoint(
    CHECKPOINT_PATH
)


# %%
title = "Chest X-Ray Classification"

description = """
Upload a chest X-ray image to classify it as CHF, Normal,
or Pneumonia using a trained ResNet50 model.
"""

article = """
**Important:** This application is an educational machine-learning
project. It is not a medical device and must not be used to provide
a medical diagnosis.
"""


# %%
demo = gr.Interface(
    fn=predict_image,

    inputs=gr.Image(
        type="pil",
        image_mode="RGB",
        sources=["upload"],
        label="Upload Chest X-Ray"
    ),

    outputs=[
        gr.Image(
            label="Uploaded Chest X-Ray"
        ),

        gr.Label(
            label="Prediction Confidence",
            num_top_classes=3
        )
    ],

    title=title,
    description=description,
    article=article,
    flagging_mode="never"
)


# %%
if __name__ == "__main__":
    demo.launch(
        inbrowser=True
    )
# %%
