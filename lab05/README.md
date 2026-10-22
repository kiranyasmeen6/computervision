# Lab 05: HOG-Based Industrial Defect Detection and Classification

**Author:** Kiran Yasmeen
**Notebook:** `Lab05_HOG_Defect_Detection.ipynb`

## Overview

This lab builds a classical computer vision pipeline that detects and classifies surface defects on steel using **Histogram of Oriented Gradients (HOG)** features and machine learning classifiers.

```
Image -> Preprocessing (grayscale, resize 128x128) -> HOG features -> Classifier -> Decision (ACCEPT / REJECT)
```

Any predicted defect type means the product is **REJECTED**.

## Dataset

- **NEU-DET** (NEU Steel Surface Defect Database), train/validation split version
- Kaggle slug: `sovitrath/neu-steel-surface-defect-detecttrainvalid-split`
- The notebook first tries Kaggle (`kagglehub`). If that fails (for example a 403 error), it automatically downloads a Google Drive copy of NEU-DET instead.
- 6 defect classes: `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`

| Split | Images per class | Total |
|---|---|---|
| Train | 240 | 1440 |
| Test | 60 | 360 |

**Note:** NEU-DET contains only defective surfaces, so this notebook does the **multi-class** version (6 defect types). If you have defect-free images, place them in `/content/normal/` before running. They are then added as an extra class called `normal`, which enables the ACCEPT decision. No normal images were used in the reported results.

## Requirements

- Python 3
- `kagglehub`, `gdown`, `numpy`, `pandas`, `opencv-python`, `scikit-image`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`

## How to Run

1. Open `Lab05_HOG_Defect_Detection.ipynb` in Google Colab or Jupyter (CPU is enough).
2. Run all cells from top to bottom.
3. The dataset downloads automatically.

## Pipeline Steps

| Step | What the notebook does |
|---|---|
| 1 | Load and inspect the dataset, build the train/test split |
| 2-3 | Convert to grayscale and resize every image to 128 x 128 |
| 4-5 | Extract HOG features and visualize them (image vs HOG picture) |
| 11 | HOG parameter study: cell size x orientations |
| 6-10 | Train and compare SVM, Random Forest and KNN on the test split |
| 12 | Robustness test with brightness, noise, rotation and blur |
| 13 | Quality-control decision module (ACCEPT / REJECT with confidence) |
| Bonus | Video-stream prototype |

## Experiments and Results

### 1. HOG parameter study

To avoid choosing parameters on test data, 20% of the training set was held out as a validation set (stratified, seed 42). A Linear SVM was used for this sweep.

| Cell size | Orientations | Features | Accuracy (%) | F1 (%) | Time (s) |
|---|---|---|---|---|---|
| 4 | 6 | 23064 | 59.72 | 59.07 | 62.0 |
| 4 | 9 | 34596 | 63.89 | 63.53 | 59.9 |
| 4 | 12 | 46128 | 72.57 | 72.30 | 56.3 |
| 8 | 6 | 5400 | 67.01 | 66.38 | 36.3 |
| 8 | 9 | 8100 | 70.14 | 69.59 | 30.6 |
| 8 | 12 | 10800 | 75.69 | 75.19 | 27.3 |
| 16 | 6 | 1176 | 77.08 | 76.88 | 13.5 |
| 16 | 9 | 1764 | 73.96 | 73.60 | 17.1 |
| 16 | 12 | 2352 | 76.04 | 75.88 | 16.9 |

**Selected setting:** cell size 16 x 16, 6 orientations (highest validation F1, and only 1176 features).

### 2. SVM hyperparameter (C)

RBF SVM, validation macro-F1: C=1 gives 0.923, C=10 gives 0.920, C=100 gives 0.913. **C = 1 was selected.**

### 3. Classifier comparison (test split)

| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Train time (s) | Inference (ms/image) |
|---|---|---|---|---|---|---|
| **SVM (RBF)** | **92.50** | 92.60 | 92.50 | **92.46** | 4.7 | 2.644 |
| Random Forest | 90.00 | 90.24 | 90.00 | 89.94 | 17.2 | 0.273 |
| KNN (k=5) | 70.28 | 77.94 | 70.28 | 67.17 | 0.0 | 0.347 |

Precision, recall and F1 are macro-averaged. SVM (RBF) was the best classifier.

Per-class F1 for the SVM: crazing 0.945, inclusion 0.932, patches 0.870, pitted_surface 0.869, rolled-in_scale 1.000, scratches 0.932.

### 4. Robustness test

The trained SVM was **not retrained**. Test images were modified and re-classified.

| Condition | Accuracy (%) | F1 (%) | Accuracy change |
|---|---|---|---|
| Clean | 92.50 | 92.46 | 0.00 |
| Brightness x0.6 | 91.11 | 90.97 | -1.39 |
| Brightness x1.4 | 67.78 | 67.66 | -24.72 |
| Gaussian noise sigma=10 | 24.44 | 17.27 | -68.06 |
| Gaussian noise sigma=25 | 18.61 | 8.23 | -73.89 |
| Rotation 10 deg | 91.39 | 91.45 | -1.11 |
| Rotation 30 deg | 75.83 | 75.76 | -16.67 |
| Blur 3x3 | 77.22 | 76.80 | -15.28 |
| Blur 7x7 | 19.44 | 10.84 | -73.06 |

### 5. Quality-control decision module

`inspect_product(image)` resizes the image, extracts HOG, predicts the class with the SVM, and prints:

- Prediction (DEFECTIVE / NON-DEFECTIVE)
- Defect type
- Confidence (%)
- Action (REJECT PRODUCT / ACCEPT PRODUCT)

If confidence is below 60%, a note recommends a manual check.

Example output: *Prediction: DEFECTIVE, Defect type: crazing, Confidence: 97%, Action: REJECT PRODUCT.*

### 6. Video-stream prototype

Processing 60 test images one after another as a frame stream ran at about **141.9 frames per second** on CPU, each frame labeled PASS or DEFECTIVE.

## Output Files

| File | Content |
|---|---|
| `hog_parameter_study.csv` | HOG cell size x orientation results |
| `classifier_comparison.csv` | SVM vs Random Forest vs KNN |
| `robustness_results.csv` | Robustness test results |
| `hog_svm_model.joblib` | Saved SVM model with HOG settings and class names |

## Discussion

- **SVM (RBF) with HOG** gave the best result (92.50% accuracy), ahead of Random Forest (90.00%) and KNN (70.28%).
- A **larger cell size** (16 x 16) worked better than smaller cells here, with far fewer features and faster extraction.
- `rolled-in_scale` was the easiest class (F1 1.000). `patches` and `pitted_surface` were the hardest (F1 about 0.87).
- HOG + SVM is **fast** (about 2.6 ms per image) and runs on CPU.

**Limitations**
- The model is **sensitive to noise, strong blur and brightness increase**. Accuracy dropped to about 19-24% under heavy noise or blur, so it needs controlled imaging conditions or noise filtering before use.
- Only defective images were available, so no real ACCEPT decision could be evaluated.
- The HOG parameter sweep used a Linear SVM, while the final model uses an RBF SVM.
- The test split is small (360 images).
