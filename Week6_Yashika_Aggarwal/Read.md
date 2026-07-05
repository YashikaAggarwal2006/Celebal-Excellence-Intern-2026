# Denoising Autoencoder on MNIST using CNN

## Project Overview

This project implements a **Convolutional Denoising Autoencoder (DAE)** using TensorFlow/Keras to remove Gaussian noise from handwritten digit images in the MNIST dataset.

The model is trained using **noisy images as input** and **clean images as target**, enabling it to reconstruct denoised images while preserving the digit structure.

---

## Objectives

- Load the MNIST dataset from the Kaggle PNG dataset.
- Preprocess the dataset by normalizing and reshaping images.
- Add artificial Gaussian noise to the images.
- Build a CNN-based Denoising Autoencoder.
- Train the model using noisy images as input and clean images as target.
- Generate denoised images from noisy test images.
- Evaluate reconstruction quality using MSE and PSNR.
- Visualize the original, noisy, and reconstructed images.

---

## Dataset

**Dataset Name:** MNIST PNG Dataset

**Source:**
https://www.kaggle.com/datasets/awsaf49/mnist-dataset

Dataset Structure:

```
mnist_png/
│
├── training/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
│
└── testing/
    ├── 0/
    ├── 1/
    ├── ...
    └── 9/
```

- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28
- Grayscale Images

---

## Data Preprocessing

The following preprocessing steps were performed:

- Loaded images from folders.
- Converted images to grayscale.
- Normalized pixel values from **0–255** to **0–1**.
- Reshaped images to **28 × 28 × 1**.
- Added Gaussian noise using NumPy.
- Clipped pixel values to remain within the range [0,1].

---

## Model Architecture

### Encoder

- Conv2D (32 filters)
- MaxPooling2D
- Conv2D (64 filters)
- MaxPooling2D

### Decoder

- Conv2D (64 filters)
- UpSampling2D
- Conv2D (32 filters)
- UpSampling2D
- Conv2D (1 filter, Sigmoid activation)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Epochs | 20 |
| Batch Size | 128 |
| Callback | EarlyStopping |

---

## Evaluation Metrics

The following metrics are used:

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)

---

## Output Files

After running the notebook, the following files are generated:

```
best_autoencoder.keras
loss_curve.png
denoising_results.png
```

---

## Results

The notebook displays:

- Original Images
- Noisy Images
- Denoised Images

It also plots:

- Training Loss
- Validation Loss

---

## Observations

- The model successfully learned to remove Gaussian noise from MNIST images.
- The reconstructed images retained the handwritten digit structure.
- CNN layers effectively captured spatial features.
- EarlyStopping reduced overfitting.
- MSE decreased steadily during training, indicating successful learning.
- PSNR values confirmed good reconstruction quality.

---

## Challenges

- Higher noise levels produced slightly blurred reconstructions.
- Increasing model complexity increased training time.
- Careful normalization was necessary for stable training.

---

## Conclusion

The Convolutional Denoising Autoencoder effectively reconstructed clean handwritten digit images from noisy inputs. The model achieved good reconstruction quality using MSE loss and the Adam optimizer while preserving the essential features of the original digits.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow (PIL)

---

