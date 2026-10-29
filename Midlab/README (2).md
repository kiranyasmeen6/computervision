# Lab Mid Exam: HOG vs SIFT on Blood-Cell Images

**Registration No:** 037 | **Name:** Kiran

**Settings:** seed = 37, HOG cell = 16 x 16 (last digit 7 is odd), SIFT grid = 6 x 6 = 36 keypoints (last digit 7 is in 5 to 9).

---

## Q1. Data setup

BloodMNIST (size 64), classes eosinophil, monocyte, neutrophil, platelet. Images converted to grayscale and resized to 128 x 128.

| Split | Shape | Eosinophil | Monocyte | Neutrophil | Platelet |
|---|---|---|---|---|---|
| Train | (400, 128, 128) | 100 | 100 | 100 | 100 |
| Validation | (120, 128, 128) | 30 | 30 | 30 | 30 |
| Test | (120, 128, 128) | 30 | 30 | 30 | 30 |

---

## Q2. HOG + SVM

**Feature length (by hand):**
- Cells per side = 128 / 16 = 8, so 8 x 8 cells
- Blocks per side = 8 - 2 + 1 = 7, so 7 x 7 = 49 blocks
- Values per block = 2 x 2 cells x 9 orientations = 36
- Total = 49 x 36 = **1764** (matches the code output)

**Choosing C (validation accuracy):**

| C | Validation accuracy |
|---|---|
| 0.1 | 0.7250 |
| 1 | 0.7333 |
| 10 | 0.7333 |

Best C = 1. **Validation accuracy = 0.7333, macro-F1 = 0.7340.**

**Confusion matrix (validation):**

| True \ Predicted | Eosinophil | Monocyte | Neutrophil | Platelet |
|---|---|---|---|---|
| Eosinophil | 17 | 9 | 4 | 0 |
| Monocyte | 7 | 20 | 3 | 0 |
| Neutrophil | 6 | 3 | 21 | 0 |
| Platelet | 0 | 0 | 0 | 30 |

---

## Q3. SIFT + SVM

**Normal SIFT keypoints (training images):** average 72.48, minimum 4, maximum 213. Images with zero keypoints: 0.

**Why a different number of keypoints is a problem for an SVM:** each keypoint gives a 128-value descriptor, so images with different numbers of keypoints produce feature vectors of different lengths, but an SVM needs every image to have the same fixed-length vector.

**Fixed grid feature length (by hand):** 6 x 6 = 36 grid points, each with 128 values, so 36 x 128 = **4608** (matches the code output).

**Choosing C (validation accuracy):**

| C | Validation accuracy |
|---|---|
| 0.1 | 0.7167 |
| 1 | 0.7250 |
| 10 | 0.7250 |

Best C = 1. **Validation accuracy = 0.7250, macro-F1 = 0.7211.**

**Confusion matrix (validation):**

| True \ Predicted | Eosinophil | Monocyte | Neutrophil | Platelet |
|---|---|---|---|---|
| Eosinophil | 15 | 7 | 7 | 1 |
| Monocyte | 9 | 18 | 3 | 0 |
| Neutrophil | 4 | 1 | 25 | 0 |
| Platelet | 0 | 0 | 1 | 29 |

---

## Q4. Robustness

**Prediction (written before running):**
- **Rotation:** HOG will lose more accuracy, because HOG stores edge-direction histograms in fixed cells, so a 20 degree turn shifts the orientation bins and moves content between cells.
- **Noise:** HOG will lose more accuracy, because HOG works directly on raw pixel gradients, which noise changes a lot, while SIFT smooths the image and looks at larger 32-pixel patches, so noise averages out.

**Test results (test set, models frozen):**

| Model | Clean accuracy | Rotated accuracy | Noisy accuracy |
|---|---|---|---|
| HOG + SVM | 0.7167 | 0.5333 | 0.2500 |
| SIFT + SVM | 0.7583 | 0.5167 | 0.7333 |

**Was my prediction correct?**

My prediction was only partly correct. For noise I predicted that HOG would lose more accuracy and this was correct: HOG dropped from 0.7167 to 0.2500 (a drop of 0.467), while SIFT dropped only from 0.7583 to 0.7333 (0.025). For rotation I also predicted HOG, but this was wrong: SIFT dropped by 0.242 (0.7583 to 0.5167) and HOG by 0.183 (0.7167 to 0.5333), so both models are sensitive to a 20 degree rotation and SIFT is slightly worse.

---

## Q5. Short answers and AI log

**(a) Why does HOG use block normalisation?** Gradient strength changes with lighting and contrast, so the same cell can give very different histogram values in different images. Normalising each block of cells makes the features less sensitive to these brightness and contrast changes, which makes the descriptor more reliable.

**(b) Why is it wrong to choose C using the test set?** The test set must only measure how the final model performs on unseen data. If C is chosen using test accuracy, the test set influences the model, so the reported accuracy becomes too optimistic. C must be chosen on the validation set, and the test set used once at the end.

**Most confused classes:** both models confused **eosinophil and monocyte** the most (9 validation images each way: HOG predicted eosinophil as monocyte, SIFT predicted monocyte as eosinophil). These cells have a similar size and round shape, and one main difference between them is colour, such as the red/orange granules of eosinophils. Grayscale removes colour, so only shape and texture remain, which makes the two classes look more alike.

**AI log**

| Prompt (one line) | What I changed | How I checked it was correct |
|---|---|---|
| Asked an AI tool to write the HOG vs SIFT notebook for this exam | Set my seed (37), HOG cell (16x16) and SIFT grid (6x6) from registration 037; wrote my own prediction in Q4 and my own explanations | Printed shapes and class counts, checked HOG length = 1764 and SIFT length = 4608 against my hand calculation, looked at the HOG, keypoint and rotated/noisy pictures |
