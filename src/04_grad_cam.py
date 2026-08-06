# %%
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn
from torchvision import models, transforms
from PIL import Image


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# %%
def load_checkpoint(filepath):
    """
    Load a ResNet50 or VGG16 checkpoint.
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

    architecture = checkpoint["arch"].lower()
    class_to_idx = checkpoint["class_to_idx"]

    num_classes = checkpoint.get(
        "num_classes",
        len(class_to_idx)
    )

    if "resnet50" in architecture:
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

        model_name = "resnet50"

    elif "vgg16" in architecture:
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

        model_name = "vgg16"

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

    loaded_model.model_name = model_name

    loaded_model = loaded_model.to(DEVICE)
    loaded_model.eval()

    print(f"Loaded: {filepath}")
    print(f"Architecture: {model_name}")

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


display_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])


# %%
def load_image(image_path):
    """
    Load one image and prepare it for the model.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = image_transform(
        original_image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    return original_image, image_tensor


# %%
class GradCAM:
    def __init__(
        self,
        model,
        target_layer
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activations
            )
        )

    def save_activations(
        self,
        module,
        inputs,
        output
    ):
        self.activations = output

        output.register_hook(
            self.save_gradients
        )

    def save_gradients(
        self,
        gradient
    ):
        self.gradients = gradient

    def generate(
        self,
        image_tensor,
        class_index=None
    ):
        """
        Generate the Grad-CAM heatmap.
        """

        self.model.zero_grad(
            set_to_none=True
        )

        output = self.model(
            image_tensor
        )

        if class_index is None:
            class_index = output.argmax(
                dim=1
            ).item()

        class_score = output[
            0,
            class_index
        ]

        class_score.backward()

        if (
            self.activations is None
            or self.gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM data was not captured."
            )

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        heatmap = (
            weights * self.activations
        ).sum(dim=1)

        heatmap = torch.relu(
            heatmap
        )[0]

        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()

        if heatmap_max > heatmap_min:
            heatmap = (
                heatmap - heatmap_min
            ) / (
                heatmap_max - heatmap_min
            )
        else:
            heatmap = torch.zeros_like(
                heatmap
            )

        heatmap = (
            heatmap.detach()
            .cpu()
            .numpy()
        )

        return heatmap, class_index, output

    def remove_hooks(self):
        self.forward_hook.remove()


# %%
def get_target_layer(model):
    """
    Select the correct final convolution layer.
    """

    if model.model_name == "resnet50":
        return model.layer4[-1]

    if model.model_name == "vgg16":
        return model.features[28]

    raise ValueError(
        "Unsupported model for Grad-CAM."
    )


# %%
def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.45
):
    """
    Overlay the heatmap on the X-ray.
    """

    display_image = display_transform(
        original_image
    )

    original_array = np.array(
        display_image
    )

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        display_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = (
        np.array(heatmap_image)
        / 255.0
    )

    colored_heatmap = plt.get_cmap(
        "turbo"
    )(heatmap_array)[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    overlay = (
        (1 - alpha) * original_array
        + alpha * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return (
        display_image,
        Image.fromarray(overlay)
    )


# %%
def generate_gradcam_result(
    model,
    image_path,
    expected_class,
    output_directory
):
    """
    Generate and save Grad-CAM for one image.
    """

    original_image, image_tensor = load_image(
        image_path
    )

    target_layer = get_target_layer(
        model
    )

    grad_cam = GradCAM(
        model=model,
        target_layer=target_layer
    )

    try:
        heatmap, predicted_index, output = (
            grad_cam.generate(
                image_tensor
            )
        )

        probabilities = torch.exp(
            output
        )

        confidence = probabilities[
            0,
            predicted_index
        ].item()

        predicted_class = model.idx_to_class[
            predicted_index
        ]

        display_image, overlay_image = (
            create_gradcam_overlay(
                original_image,
                heatmap
            )
        )

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        file_prefix = (
            f"{model.model_name}_"
            f"{expected_class.lower()}"
        )

        comparison_path = (
            output_directory
            / f"{file_prefix}_comparison.png"
        )

        overlay_path = (
            output_directory
            / f"{file_prefix}_overlay.png"
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        axes[0].imshow(
            display_image
        )

        axes[0].set_title(
            f"Original: {expected_class}"
        )

        axes[0].axis("off")

        axes[1].imshow(
            heatmap,
            cmap="turbo"
        )

        axes[1].set_title(
            "Grad-CAM Heatmap"
        )

        axes[1].axis("off")

        axes[2].imshow(
            overlay_image
        )

        axes[2].set_title(
            f"Predicted: {predicted_class}\n"
            f"Confidence: {confidence:.2%}"
        )

        axes[2].axis("off")

        plt.suptitle(
            f"{model.model_name.upper()} "
            f"Grad-CAM — {expected_class}"
        )

        plt.tight_layout()

        plt.savefig(
            comparison_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        overlay_image.save(
            overlay_path
        )

        print()
        print(f"Model: {model.model_name}")
        print(f"Actual class: {expected_class}")
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Saved: {comparison_path}")
        print(f"Saved: {overlay_path}")

    finally:
        grad_cam.remove_hooks()


# %%
resnet_checkpoint = Path(
    "../checkpoint/20260727_resnet50.pth"
)

vgg_checkpoint = Path(
    "../checkpoint/20260727_vgg16.pth"
)


# Change these filenames to real images in your folders.
image_paths = {
    "CHF": Path(
        "../img/Dataset/CHF/"
        "010fa20c-6ac04c8a-f6d4bc0b-eb1e735c-cd940793.jpeg"
    ),

    "Normal": Path(
        "../img/Dataset/Normal/"
        "0a90c3bb-3cf21ec9-835a1537-faf370a1-21053fd6.jpeg"
    ),

    "Pneumonia": Path(
        "../img/Dataset/Pneumonia/"
        "00fe73b4-5215bb4f-94bbccc4-ac5f4f6f-52805cfb.jpeg"
    )
}


output_directory = Path(
    "../plots/gradcam"
)


# %%
resnet_model = load_checkpoint(
    resnet_checkpoint
)

for class_name, image_path in image_paths.items():
    generate_gradcam_result(
        model=resnet_model,
        image_path=image_path,
        expected_class=class_name,
        output_directory=output_directory
    )

del resnet_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %%
vgg_model = load_checkpoint(
    vgg_checkpoint
)

for class_name, image_path in image_paths.items():
    generate_gradcam_result(
        model=vgg_model,
        image_path=image_path,
        expected_class=class_name,
        output_directory=output_directory
    )

del vgg_model

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# %%
print("All six Grad-CAM results were created.")