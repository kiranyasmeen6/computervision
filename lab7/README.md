# Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm

**Author:** Kiran Yasmeen
**Notebook:** `Lab07_GA_HOG_Feature_Selection.ipynb`

## Research Question

Can Genetic Algorithm (GA)-based feature selection improve the classification performance of HOG features while reducing feature dimensionality on medical images?

## Overview

This lab modifies the traditional HOG + SVM pipeline by adding a Genetic Algorithm that selects a subset of HOG features. The baseline (HOG + SVM, all features) is compared with the proposed model (GA-HOG + SVM, selected features) on the same fixed data split.

```
Images -> Grayscale + Resize (128x128) -> HOG features (8100)
       -> Baseline SVM (all features)
       -> Genetic Algorithm (feature selection)
       -> SVM on selected features
       -> Compare on test set
```

## Dataset

- **BloodMNIST** (MedMNIST v2): 8 classes of blood cell images
- Source: https://medmnist.com/
- References:
  - Yang et al., "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification", *Scientific Data*, 2023.
  - Acevedo et al., "A dataset of microscopic peripheral blood cell images for development of automatic recognition systems", *Data in Brief*, 2020.

Balanced subset used (same split for all models, fixed seed 42):

| Split | Images per class | Total |
|---|---|---|
| Train | 200 | 1600 |
| Validation | 100 | 800 |
| Test | 200 | 1600 |

## Requirements

- Python 3
- `medmnist`, `numpy`, `pandas`, `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`, `seaborn`

Install the dataset package (done in the first notebook cell):

```
pip install medmnist
```

## How to Run

1. Open `Lab07_GA_HOG_Feature_Selection.ipynb` in Google Colab or Jupyter (CPU is enough).
2. Run all cells from top to bottom.
3. The dataset downloads automatically on first run.
4. The GA search takes several minutes (about 8 minutes for all three GA settings in our run).
5. The test set is used only in the final evaluation cells, after model selection is complete.

## Configuration

| Parameter | Baseline: HOG + SVM | Proposed: GA-HOG + SVM |
|---|---|---|
| Dataset | BloodMNIST | BloodMNIST |
| Image resolution | 128 x 128 | 128 x 128 |
| HOG orientations | 9 | 9 |
| Pixels per cell | 8 x 8 | 8 x 8 |
| Cells per block | 2 x 2 | 2 x 2 |
| Block normalization | L2-Hys | L2-Hys |
| Number of features | 8100 | Selected by GA |
| Feature selection | None | Genetic Algorithm |
| Classifier | SVM (RBF kernel) | SVM (RBF kernel) |
| Hyperparameter tuning | C in {1, 10, 100} on validation set | C in {1, 10, 100} on validation set |
| Data split | Identical fixed split | Identical fixed split |
| Random seed | 42 | 42 |

## Genetic Algorithm

Each individual is a binary vector of length 8100 (1 = keep the feature, 0 = remove it).

| Component | Setting |
|---|---|
| Initial population | Random 0/1 vectors |
| Fitness | `validation macro-F1 - 0.05 x (selected features / all features)` |
| Selection | Tournament selection (size 3) |
| Crossover | Uniform crossover, probability 0.9 |
| Mutation | Bit flip, probability 0.005 or 0.02 |
| Elitism | Top 2 individuals kept unchanged |
| Generations | 15 |

The fitness evaluation uses a precomputed RBF kernel for speed. A check in the notebook confirms its predictions are 100% identical to the standard `SVC(kernel="rbf")`.

### GA settings tested

| Setting | Population | Mutation | Best fitness | Val macro-F1 | Features kept | GA time |
|---|---|---|---|---|---|---|
| GA-1 | 20 | 0.005 | 0.6683 | 0.6933 | 4040 | 113 s |
| **GA-2 (selected)** | **40** | **0.005** | **0.6894** | **0.7145** | **4075** | **238 s** |
| GA-3 | 20 | 0.02 | 0.6783 | 0.7032 | 4022 | 109 s |
| Baseline | - | - | - | 0.6604 | 8100 | - |

GA-2 was selected using validation fitness only.

## Results (Test Set)

| Metric | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) | Improvement |
|---|---|---|---|
| Accuracy (%) | 66.69 | 66.69 | 0.00 |
| Precision (Macro, %) | 67.14 | 66.95 | -0.19 |
| Recall (Macro, %) | 66.69 | 66.69 | 0.00 |
| Macro-F1 (%) | 66.66 | 66.65 | -0.02 |
| Weighted-F1 (%) | 66.66 | 66.65 | -0.02 |
| Balanced accuracy (%) | 66.69 | 66.69 | 0.00 |
| Specificity (Macro, %) | 95.24 | 95.24 | 0.00 |
| Cohen's kappa | 0.6193 | 0.6193 | 0.0000 |
| Matthews correlation coefficient | 0.6199 | 0.6197 | -0.0002 |

## Computational Cost

| Item | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) |
|---|---|---|
| Number of HOG features | 8100 | 4075 |
| Feature reduction | 0% | 49.7% |
| HOG extraction time (s) | 27.2 | 27.2 |
| GA search time (s) | 0 | 238.0 |
| SVM training time (s) | 15.6 | 6.2 |
| Inference time (ms per image) | 12.118 | 7.227 |
| Total pipeline time (s) | 42.8 | 271.4 |
| Validation macro-F1 | 0.6604 | 0.7145 |

Timing values depend on the machine and may vary between runs.

## Plots Produced

1. Sample image from each of the 8 classes
2. GA fitness across generations (best and mean, all three settings)
3. Number of features kept per generation
4. Test performance comparison (baseline vs GA-HOG)
5. Features before and after selection
6. Heatmap of where the GA kept features across image blocks
7. Confusion matrices for both models

## Output Files

| File | Content |
|---|---|
| `table1_configuration.csv` | Experimental configuration |
| `table2_main_comparison.csv` | Main performance comparison |
| `table3_computational_cost.csv` | Computational cost comparison |
| `table_ga_settings.csv` | Results of the three GA settings |

## Discussion

**Findings**
- The GA removed about half of the HOG features (8100 to 4075).
- Test performance stayed essentially unchanged (accuracy 66.69% for both models), so GA did not improve accuracy on the test set.
- The reduced model trains faster (15.6 s to 6.2 s) and predicts faster (12.1 to 7.2 ms per image).
- The GA fitness increased steadily over generations, and the larger population (GA-2) reached the highest fitness.

**Strengths**
- Same accuracy with a feature set about half the size.
- Faster SVM training and inference.

**Limitations**
- The GA search is expensive: an SVM is trained for each individual in each generation (238 s for the selected setting, making the total pipeline much longer than the baseline).
- The validation macro-F1 improved (0.6604 to 0.7145), but this did not carry over to the test set, which suggests the GA fitted somewhat to the validation data.
- The number of selected features stayed near 50% from the first generation, because the random initial population keeps about half of the features. The penalty alone did not push it lower within 15 generations.
- Results depend on the random seed and on the reduced subset of the data used.

**Conclusion:** GA-based feature selection reduced feature dimensionality by about 50% without reducing test performance, but it did not improve classification accuracy on BloodMNIST. It also adds a one-time search cost.
