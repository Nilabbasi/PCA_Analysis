# PCA Analysis on the MNIST Dataset

---

## Introduction

This repository contains an implementation of Principal Component Analysis (PCA) on the MNIST dataset. 
PCA is a powerful technique for dimensionality reduction, which helps simplify datasets while preserving essential information.
The code is thoroughly commented with single-line and multi-line explanations to ensure clarity and understanding.

## Table of Contents
- [1. Required Libraries](#1-required-libraries)
- [2. Loading and Preprocessing the MNIST Dataset](#2-loading-and-preprocessing-the-mnist-dataset)
- [3. Normalization and Covariance Calculation](#3-normalization-and-covariance-calculation)
- [4. Principal Component Extraction](#4-principal-component-extraction)
- [5. Dimensionality Reduction and Reconstruction](#5-dimensionality-reduction-and-reconstruction)
- [6. Model Training and Evaluation](#6-model-training-and-evaluation)
- [7. Results](#7-results)

---

## 1. Required Libraries

The following libraries are required for this analysis. Ensure they are installed in your environment:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from prettytable import PrettyTable
from keras.datasets import mnist

## 2. Loading and Preprocessing the MNIST Dataset

We load the MNIST dataset and check for any NaN values. If found, they are replaced with zeros. The data is then reshaped into 784-dimensional vectors for further processing.

## 3. Normalization and Covariance Calculation

To prepare the data for PCA, we scale it to the range [0, 1] and compute the covariance matrix. The covariance matrix can be calculated both manually and using built-in functions.

### Manual Calculation of Covariance Matrix

1. **Normalize the Data**: Scale the dataset to the range [0, 1].
2. **Calculate the Covariance Matrix Manually**: 
   - Center the data by subtracting the mean.
   - Compute the covariance matrix using the centered data.

### Automatic Calculation of Covariance Matrix

We also provide a method to calculate the covariance matrix automatically using NumPy's built-in functions.

## 4. Principal Component Extraction

We extract the eigenvalues and eigenvectors of the covariance matrix to determine the principal components that explain a satisfactory portion of the variance in the dataset.

## 5. Dimensionality Reduction and Reconstruction

We project the original data into a reduced dimensionality space and then reconstruct the images to visualize the effect of dimensionality reduction.

## 6. Model Training and Evaluation

We build a **logistic regression** model and evaluate its accuracy based on the number of principal components used.

## 7. Results

The results include accuracy metrics based on varying the number of principal components, which are presented in a table format in the notebook.

### Accuracy vs. Number of Components Plot

Below is a plot illustrating the relationship between the number of principal components and the model's accuracy:

![Accuracy vs. Number of Components](path/to/your/image.png)

---

## Conclusion

This repository demonstrates the implementation of PCA on the MNIST dataset, showcasing its effectiveness in reducing dimensionality while maintaining essential information for classification tasks.

