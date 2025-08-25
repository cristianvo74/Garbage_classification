# Automatic Waste Classification System Using Convolutional Neural Networks

## Overview
This project aims to develop an intelligent system capable of automatically classifying different types of municipal solid waste using computer vision and deep learning techniques. The system employs a Convolutional Neural Network (ResNet-50) trained on a public dataset of garbage images, enabling it to identify seven main categories of recyclable materials.

The solution includes:
- A reproducible Jupyter Notebook for model training and evaluation.
- A script for real-time deployment using a webcam.
- Documentation and configuration files to facilitate installation and usage.

## Motivation
Efficient waste management is crucial for environmental sustainability. Automating garbage classification reduces human error, improves recycling rates, and helps decrease cross-contamination between materials.

## Project Structure

- `Classifier_training.ipynb`: Main notebook for model training and evaluation.
- `realtime_classifier.py`: Script for real-time classification using a webcam.
- `propuesta_proyecto_garbage_classification.md`: Project proposal and justification document.
- `requirements.txt`: List of required dependencies.
- `.gitignore`: Files and folders excluded from the repository.
- `garbage_classification/`: Folder containing the dataset images (not included in the repository due to its size).
- `mejor_modelo.pth` & `resnet50_Garbage_final.pth`: Trained model weights (not included in the repository).

## System and Library Requirements

- Python 3.8 or higher (3.11 recommended)
- Operating System: Linux, Windows, or MacOS
- GPU is optional for accelerating the training process.

### Main Libraries:
- torch
- torchvision
- opencv-python
- Pillow
- numpy
- scikit-learn
- matplotlib
- tqdm

## Dependency Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Joseph0choa/Garbage.git](https://github.com/Joseph0choa/Garbage.git)
   cd Garbage
   ```
2. (Optional) Create and activate a conda environment:

```Bash
conda create -n Garbage python=3.11
conda activate Garbage 
```
3. Install the dependencies:
```Bash
pip install -r requirements.txt
```
## Step-by-Step Guide to Train the Model

1.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) and place it in the `garbage_classification/` folder.
2.  Open and run the `Classifier_training.ipynb` notebook to:
    - Preprocess the data
    - Define and train the ResNet-50 model
    - Validate and evaluate its performance
    - Save the trained model (`mejor_modelo.pth`)

---

## How to Load the Trained Model

You can load the model in any script as follows:

```python
import torch
from torchvision import models
import torch.nn as nn

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7) # 7 classes
model.load_state_dict(torch.load('mejor_modelo.pth', map_location='cpu'))
model.eval()
```

## Real-Time Deployment Example (Webcam)

1.  Run the script:
    ```bash
    python realtime_classifier.py
    ```
2.  A window will open, displaying the real-time prediction on the image captured by the webcam. The text color will correspond to the detected category.
3.  Press `ESC` to close the window.

---

## Authors / Credits

**Cristian Andres Villarreal Orozco** **Anderson Joseph Ochoa Trujillo**

Project for the **Neural Networks and Deep Learning** course.
