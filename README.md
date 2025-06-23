## For most up to date work it is on Colab https://drive.google.com/file/d/1gRotJy-v4wtWBNArpwyxX9gEiw6S17ZO/view?usp=sharing
# Project 1: Advanced Image Classification using CNNs and Transfer Learning

## 1. Project Goal
The objective of this project is to build and evaluate deep learning models for multi-class image classification. We will tackle the Intel Image Classification dataset, which contains natural scenes from around the world. I chose this set cause it was one of the more popular and relevant sets on Kaggle.

This project demonstrates proficiency in:
- Building a Convolutional Neural Network (CNN) from scratch
- Implementing a robust data pipeline with data augmentation
- Applying advanced techniques like Transfer Learning (using EfficientNetB0) and Fine-Tuning
- Utilizing callbacks for efficient training (`EarlyStopping`, `ModelCheckpoint`)
- Comprehensive model evaluation using accuracy, confusion matrices, and classification reports
- Understanding and mitigating overfitting through regularization and dropout

## 2. Dataset
- **Source:** [Intel Image Classification on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Description:** The dataset contains ~14,000 training images, ~3,000 validation images, and ~7,000 testing images, categorized into 6 classes: 'buildings', 'forest', 'glacier', 'mountain', 'sea', 'street'.
- **Image Size:** Images are 150x150 pixels.

## 3. Models Implemented

### Model 1: CNN from Scratch
A custom-built CNN to establish a baseline performance. The architecture consists of:
- Multiple `Conv2D` layers with `ReLU` activation to learn spatial features.
- `MaxPooling2D` layers for down-sampling and spatial invariance.
- `Dropout` layers to prevent overfitting.
- A `Flatten` layer followed by `Dense` layers for classification.

### Model 2: Transfer Learning with EfficientNetB0
This model leverages a state-of-the-art pre-trained model to achieve higher accuracy.
- **Feature Extraction:** The convolutional base of `EfficientNetB0` (pre-trained on ImageNet) is used as a feature extractor. Its weights are initially frozen.
- **Custom Head:** A new classification head (a `GlobalAveragePooling2D` layer and `Dense` layers) is added on top of the frozen base.
- **Fine-Tuning:** After initial training, the top layers of the EfficientNetB0 base are unfrozen and the entire model is trained with a very low learning rate to fine-tune the pre-trained weights to our specific dataset.

## 4. Key Steps in the Notebook
1.  **Setup & Data Loading:** Import necessary libraries and load the dataset using `tf.keras.utils.image_dataset_from_directory`.
2.  **Data Exploration & Visualization:** Display sample images and their labels.
3.  **Data Augmentation:** A sequential data augmentation layer is created to apply random flips, rotations, and zooms during training, making the model more robust.
4.  **Model Building:** Implementation of the two models described above.
5.  **Training:** Each model is compiled with an `Adam` optimizer and `SparseCategoricalCrossentropy` loss. Training is performed using `model.fit()`, with `EarlyStopping` and `ModelCheckpoint` callbacks.
6.  **Evaluation:** Models are evaluated on the test set. Performance is analyzed using loss/accuracy plots, a confusion matrix, and a detailed classification report.
7.  **Comparison & Conclusion:** The performance of the scratch model is compared against the transfer learning model, highlighting the significant improvement gained from leveraging pre-trained weights.

## 5. Results
| Model                       | Test Accuracy | Test Loss | Key Observation                                  |
|-----------------------------|---------------|-----------|--------------------------------------------------|
| CNN from Scratch            | ~85-88%       | ~0.45     | Decent performance but struggles with generalization. |
| Transfer Learning (Fine-Tuned) | **~93-95%**   | **~0.21** | Significantly higher accuracy and better generalization. |

The results clearly demonstrate the power of transfer learning. By using features learned from the massive ImageNet dataset, the `EfficientNetB0` model can achieve superior performance with less training time and data compared to a model trained from scratch.
