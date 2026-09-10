# Skin Cancer (ISIC) 4-Class Classification — Model Comparison

Comparison of transfer-learning CNN models, classical ML classifiers on deep features, and computational efficiency for 4-class skin lesion classification on the [ISIC Skin Cancer 9-Classes dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic).

## Table 1. Comparison of Transfer Learning Models

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|
| AlexNet | 72.73% | 77.50% | 81.25% | 70.42% | 96.28% |
| VGG16 | 81.82% | 87.50% | 87.50% | 84.52% | 97.32% |
| VGG19 | 72.73% | 83.7% | 77.0% | 75.6% | 94.4% |
| ResNet18 | 81.82% | 90.00% | 87.83% | 85.42% | 89.73% |
| ResNet50 | 63.64% | 72.50% | 72.92% | 70.83% | 86.61% |
| ResNet101 | 63.64% | 68.75% | 72.92% | 69.05% | 82.1% |
| DenseNet121 | 63.64% | 72.50% | 72.92% | 70.83% | 83.63% |
| EfficientNet-B0 | 63.64% | 56.25% | 75.00% | 63.10% | 91.52% |

## Table 2. Comparison of Different Classifiers

Classical ML classifiers trained on deep features extracted from a pretrained ResNet18.

| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Deep Features | Logistic Regression | 84.62% | 85.42% | 85.42% | 85.42% | 90.83% |
| Deep Features | Decision Tree | 53.85% | 45.00% | 54.17% | 48.89% | 69.39% |
| Deep Features | Random Forest | 76.92% | 80.42% | 77.08% | 73.47% | 91.36% |
| Deep Features | K-Nearest Neighbors (KNN) | 69.23% | 72.92% | 72.92% | 72.32% | 87.29% |
| Deep Features | Linear SVM | 84.62% | 85.42% | 85.42% | 85.42% | 91.67% |
| Deep Features | RBF-SVM | 69.23% | 70.00% | 72.92% | 69.84% | 84.86% |
| Deep Features | XGBoost | 61.54% | 50.00% | 54.17% | 50.95% | 85.40% |

## Table 3. Computational Efficiency Comparison

| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
|---|---|---|---|---|---|
| AlexNet | 57.02 | 217.51 | 1.42 | 2.10 | 72.73% |
| VGG16 | 134.28 | 512.23 | 30.93 | 8.84 | 81.82% |
| VGG19 | 139.59 | 532.48 | 39.26 | 10.51 | 72.73% |
| ResNet18 | 11.18 | 42.64 | 3.65 | 2.40 | 81.82% |
| ResNet50 | 23.52 | 89.71 | 8.26 | 8.53 | 63.64% |
| DenseNet121 | 6.96 | 26.54 | 5.79 | 21.49 | 63.64% |
| EfficientNet-B0 | 4.01 | 15.31 | 0.83 | 9.02 | 63.64% |
