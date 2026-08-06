# Chest X-Ray Disease Classification using Deep Learning

## Overview

This project implements a deep learning pipeline for automated chest X-ray disease classification using transfer learning with ResNet50 and VGG16. The models classify chest X-ray images into three categories:

- Congestive Heart Failure (CHF)
- Normal
- Pneumonia

The project includes image cleaning, data augmentation, model training, evaluation, and Grad-CAM visualizations to improve model interpretability. Training and evaluation were performed using PyTorch, and the models were developed and tested on a GPU-enabled High Performance Computing (HPC) cluster.

---

## Project Workflow

The source code is organized into sequential Python scripts that follow the complete machine learning pipeline.

- **01_image_cleaning.py**
  - Detects duplicate images and removes poor-quality samples to improve dataset quality.

- **02_image_augmentation.py**
  - Performs image augmentation and creates training, validation, and testing datasets.

- **03_image_classification.py**
  - Trains and evaluates ResNet50 and VGG16 using transfer learning.

- **04_gradcam.py**
  - Generates Grad-CAM visualizations to highlight image regions used for model predictions.

- **05_model_hosting.py**
  - Hosts the trained model for interactive prediction and visualization.

---

## Features

- Chest X-ray disease classification
- Image preprocessing and duplicate detection
- Data augmentation
- Transfer learning with ResNet50
- Transfer learning with VGG16
- Model comparison
- Confusion matrices
- Training and validation curves
- Grad-CAM visualizations
- GPU acceleration using an HPC cluster

---

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- Linux
- Slurm
- CUDA

---

## How to Run

1. Clone this repository.

```bash
git clone https://github.com/yourusername/Medical-Imaging-XRay-Classification.git
cd Medical-Imaging-XRay-Classification
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Download the chest X-ray dataset.

> **Note:** The dataset is not included in this repository due to licensing restrictions.

4. Place the dataset inside

```
img/Dataset/
```

5. Run the pipeline in order.

```
01_image_cleaning.py
02_image_augmentation.py
03_image_classification.py
04_gradcam.py
05_model_hosting.py
```

---

## Repository Structure

```text
Medical-Imaging-XRay-Classification
│
├── src/
│   ├── 01_image_cleaning.py
│   ├── 02_image_augmentation.py
│   ├── 03_image_classification.py
│   ├── 04_gradcam.py
│   └── 05_model_hosting.py
│
├── plots/
│   ├── data_analysis/
│   ├── training/
│   ├── evaluation/
│   └── gradcam/
│
├── img/
│   ├── duplicate_img_path.json
│   └── remove_img_path.json
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Future Improvements

- Train on larger chest X-ray datasets
- Evaluate additional CNN architectures
- Hyperparameter optimization
- Cross-validation
- Clinical decision support integration

---

## Acknowledgements

This project was adapted from an educational image classification workflow and expanded into a medical imaging application using chest X-ray datasets, transfer learning, and explainable AI techniques.