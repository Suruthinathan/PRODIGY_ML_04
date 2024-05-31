# Gesture Recognition with Convolutional Neural Networks
This repository contains a project for recognizing hand gestures using Convolutional Neural Networks (CNNs) implemented in TensorFlow and Keras. The model is trained on the Leap GestRecog dataset, which consists of various hand gestures captured in grayscale images.The dataset used is from [Hand Gesture Recognition Database](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).

# Model Architecture
The model is a simple Convolutional Neural Network (CNN) with the following architecture:

- Input layer: 64x64 grayscale images
+ 2 Convolutional layers with ReLU activation and MaxPooling
* Flatten layer
- Dense layer with ReLU activation
+ Output layer with Softmax activation

# Dataset
The LeapGestRecog dataset is used for training and testing the model. The dataset contains images of hand gestures performed by multiple subjects. The gestures are categorized into 10 classes: 10_down, 09_c, 08_palm_moved, 07_ok, 06_index, 05_thumb, 04_fist_moved, 03_fist, 02_l, 01_palm.

# Prerequisites
Ensure you have the following libraries installed:

+ TensorFlow
- Keras
* OpenCV
+ NumPy
- scikit-learn
+ seaborn
* matplotlib
  
You can install these libraries using pip:
```
pip install tensorflow opencv-python numpy scikit-learn seaborn matplotlib

```

# Results
The model is trained for 10 epochs with a validation split of 10%. The training and validation accuracy and loss are recorded, and the final test accuracy is evaluated on the test set.

# Training and Validation Accuracy
Epoch 1: accuracy: 0.9080, val_accuracy: 0.9856
Epoch 2: accuracy: 0.9969, val_accuracy: 0.9975
...
Epoch 10: accuracy: 1.0000, val_accuracy: 0.9994
Test Accuracy
Test accuracy: 1.0

# Confusion Matrix
The confusion matrix for the test set predictions is plotted and displayed using seaborn.

# Classification Report
A classification report with precision, recall, and F1-score for each gesture class is generated and displayed.

# Sample Predictions
The script prints sample predictions for the first 10 test images, showing the true label and the predicted label.

# Acknowledgements
The dataset used in this project is the Leap GestRecog dataset from [Hand Gesture Recognition Database](https://www.kaggle.com/datasets/gti-upm/leapgestrecog).

