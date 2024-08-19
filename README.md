# Leaf Disease Classification

This project implements a Convolutional Neural Network (CNN) to classify diseases in various types of leaves. The model is trained on a dataset of leaf images categorized by disease type and helps in the early identification and treatment of plant diseases.

## Project Overview

The goal of this project is to build a machine learning model that can accurately classify leaf diseases. The CNN model processes images of leaves and categorizes them into disease types or healthy states, aiding in agricultural diagnostics.

## Dataset

The dataset should be organized into the following structure:

dataset/
**train/**
disease_1/
disease_2/
...
healthy/
**test/**
disease_1/
disease_2/
...
healthy/


- `train/`: Contains subdirectories for each disease class, filled with corresponding training images.
- `test/`: Contains subdirectories for each disease class, filled with corresponding testing images.

Update the `train_dir` and `test_dir` variables in the script to point to your dataset's location.

## Model Architecture

The CNN model includes the following layers:
- **Convolutional Layers**: Three Conv2D layers with ReLU activation followed by MaxPooling2D layers to reduce dimensionality.
- **Flatten Layer**: Converts 2D feature maps into a 1D vector.
- **Dense Layers**: Fully connected layers with ReLU activation, followed by a Dropout layer to prevent overfitting.
- **Output Layer**: A Dense layer with `softmax` activation for multi-class classification.

## Training

- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Epochs**: 30 (modifiable)
- **Batch Size**: 32
- **Data Augmentation**: Techniques like rescaling, shearing, zooming, and horizontal flipping are applied.

The model is trained on 80% of the data, with 20% reserved for validation.

## Evaluation

After training, the model is evaluated on the test dataset. Key metrics include:
- **Confusion Matrix**: Displays the model's performance across different classes.
- **Overall Accuracy**: The percentage of correctly classified instances.

## Usage

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- PIL
