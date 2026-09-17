# Lab Task 02 — Results Table & Questions

## Required Experimental Results

| Model | Filter | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | Macro-F1 (%) | AUC (%) |
|---|---|---|---|---|---|---|---|
| Best Model 1 (VGG16) | No Filter | 59.33 | 58.26 | 59.33 | 57.55 | 57.64 | 85.15 |
| Best Model 1 (VGG16) | Average | 60.67 | 64.69 | 60.67 | 58.02 | 58.10 | 86.80 |
| Best Model 1 (VGG16) | Gaussian | 63.33 | 65.24 | 63.33 | 62.31 | 62.31 | 86.39 |
| Best Model 1 (VGG16) | Median | 60.67 | 63.71 | 60.67 | 61.42 | 61.46 | 86.60 |
| Best Model 1 (VGG16) | Sharpening | 67.33 | 66.71 | 67.33 | 65.65 | 65.63 | 88.85 |
| Best Model 1 (VGG16) | Sobel | 48.67 | 50.64 | 48.67 | 47.41 | 47.49 | 77.54 |
| Best Model 2 (ResNet18) | No Filter | 67.33 | 67.59 | 67.33 | 66.86 | 66.88 | 89.27 |
| Best Model 2 (ResNet18) | Average | 66.00 | 65.78 | 66.00 | 65.41 | 65.46 | 88.43 |
| Best Model 2 (ResNet18) | Gaussian | 70.00 | 69.78 | 70.00 | 69.64 | 69.64 | 89.69 |
| Best Model 2 (ResNet18) | Median | 66.00 | 65.90 | 66.00 | 64.57 | 64.57 | 89.65 |
| Best Model 2 (ResNet18) | Sharpening | 65.33 | 65.08 | 65.33 | 64.65 | 64.68 | 90.05 |
| Best Model 2 (ResNet18) | Sobel | 55.33 | 55.19 | 55.33 | 54.82 | 54.88 | 77.66 |
| Best Model 3 (AlexNet) | No Filter | 68.00 | 67.29 | 68.00 | 66.42 | 66.40 | 89.54 |
| Best Model 3 (AlexNet) | Average | 63.33 | 62.97 | 63.33 | 62.25 | 62.32 | 87.65 |
| Best Model 3 (AlexNet) | Gaussian | 69.33 | 70.98 | 69.33 | 68.84 | 68.91 | 89.87 |
| Best Model 3 (AlexNet) | Median | 67.33 | 67.26 | 67.33 | 66.07 | 66.09 | 88.54 |
| Best Model 3 (AlexNet) | Sharpening | 66.67 | 67.29 | 66.67 | 65.66 | 65.68 | 87.54 |
| Best Model 3 (AlexNet) | Sobel | 51.33 | 50.22 | 51.33 | 50.19 | 50.25 | 74.31 |

## Questions to Answer

**1. Which three pretrained models performed best in Lab Activity 1?**

Based on Task 01's Table 1 results (Accuracy, with AUC as tiebreaker):
1. **VGG16** — 81.82% Accuracy, 97.32% AUC
2. **ResNet18** — 81.82% Accuracy, 89.73% AUC
3. **AlexNet** — 72.73% Accuracy, 96.28% AUC (beats VGG19's 94.4% AUC at the same accuracy)

**2. How does filtering affect each of the three models?**

- **VGG16** is the most filter-sensitive in a *positive* direction — every filter except Sobel improves it over baseline, and Sharpening gives it the single biggest gain of the whole experiment (+8.00% accuracy, 57.64% → 65.63% Macro-F1). Sobel is catastrophic for it (−10.66% accuracy).
- **ResNet18** starts as the strongest baseline of the untouched runs by AUC and is fairly filter-resistant: Average, Median, and Sharpening all cause small dips (−1.3 to −2.0%), Gaussian gives its best result of all (70.00% accuracy, 69.64% Macro-F1), and Sobel again causes the largest drop (−12.00%).
- **AlexNet** also starts strong (68.00% baseline) but is the most negatively affected overall — Average, Median, and Sharpening all reduce its performance, only Gaussian helps (+1.33%), and Sobel is by far its worst result (−16.67% accuracy, the single biggest degradation of any model/filter combination).

**3. Which filter produces the greatest change compared with the unfiltered baseline?**

**Sobel** produces the largest change for all three models, and it is always a large *decrease*:
- VGG16: Sobel (−10.66% accuracy)
- ResNet18: Sobel (−12.00% accuracy)
- AlexNet: Sobel (−16.67% accuracy — the largest single change in the whole experiment)

**4. Does the effect of a filter remain consistent across all three models?**

No, only partially:

| Filter | AlexNet Δ | ResNet18 Δ | VGG16 Δ | Consistency |
|---|---|---|---|---|
| Average | −4.67 | −1.33 | +1.34 | Inconsistent |
| Gaussian | +1.33 | +2.67 | +4.00 | **Consistent** (all positive) |
| Median | −0.67 | −1.33 | +1.34 | Inconsistent |
| Sharpening | −1.33 | −2.00 | +8.00 | Inconsistent |
| Sobel | −16.67 | −12.00 | −10.66 | **Consistent** (all negative) |

Only **Gaussian** (always helps) and **Sobel** (always hurts, severely) act consistently across all three architectures. Average, Median, and Sharpening have model-dependent effects — notably Sharpening is VGG16's best filter but AlexNet's and ResNet18's worst-but-one.

**5. Does filtering improve or decrease macro-F1 and balanced accuracy?**

Averaged across all three models, Macro-F1 change vs. baseline per filter:

| Filter | Avg Macro-F1 Δ (%) |
|---|---|
| Average | −1.68 |
| Gaussian | **+3.31** |
| Median | +0.40 |
| Sharpening | +1.69 |
| Sobel | **−12.77** |

On average, **Gaussian and Sharpening improve Macro-F1**, Median is roughly neutral, Average is mildly harmful, and **Sobel is sharply harmful** — by a wide margin the worst filter overall.

**6. Which lesion classes are most affected by filtering?**

Summing the absolute per-class F1 change across all 3 models × 5 filters:

| Class | Total absolute F1 volatility |
|---|---|
| **bkl** (benign keratosis) | **173.4** — most affected |
| mel (melanoma) | 98.5 |
| bcc (basal cell carcinoma) | 84.0 |
| nv (melanocytic nevi) | 78.1 — least affected |

**bkl is by far the most filter-sensitive class** — e.g. Sobel drops its F1 by 25–31 points depending on the model, while Sharpening and Gaussian *boost* it by 16–21 points on VGG16. **nv**, the majority/easiest class, is the most stable across filters.

**7. Why might smoothing remove useful lesion texture or morphological information?**

Average, Gaussian, and Median filters all work by averaging or ranking pixel values within a local neighborhood. Skin lesions are often distinguished by fine-grained texture (e.g. pigment network, dots/globules, irregular borders) — exactly the kind of high-frequency detail that smoothing filters suppress. By blurring these details, smoothing can erase the subtle morphological cues (border irregularity, texture heterogeneity) that a CNN relies on to distinguish, say, melanoma from a benign nevus. This matches the results above: Average and Median gave AlexNet and ResNet18 small-to-moderate accuracy drops, consistent with some texture information being lost.

**8. Why might sharpening or edge detection help or hurt classification?**

Sharpening amplifies high-frequency content (edges, texture), which can help emphasize lesion borders and surface texture — potentially useful since border irregularity is a clinically relevant feature. This is exactly what happened for VGG16 (+8.00% accuracy, its best result). But sharpening can also amplify noise and artifacts (hair, lighting variation), which is likely why it *hurt* AlexNet and ResNet18 instead. Sobel goes further and discards color/intensity information entirely, keeping only edge magnitude — it was the worst filter for every model here (−10.66% to −16.67% accuracy), showing that for these architectures, color and shading cues matter more than pure edge/shape information; throwing color away removed more diagnostic signal than the edge information could recover.

**9. What is the difference between convolution and correlation?**

Both operations slide a kernel over an image and compute a weighted sum, but they differ in kernel orientation: **cross-correlation** slides the kernel as-is, while **convolution** first flips the kernel 180° (both horizontally and vertically) before sliding it. For a symmetric kernel (like a Gaussian or mean filter), the two are identical. For an asymmetric kernel (like a Sobel operator), they give different results. In practice, most deep learning frameworks (including PyTorch's `nn.Conv2d`) implement cross-correlation but call it "convolution," since the kernel weights are learned anyway and the flip makes no practical difference for a trained network.

**10. Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.**

Classical filters and deep CNN feature extraction both operate on the same core idea — convolving a kernel across an image — but classical filters use fixed, hand-designed kernels (mean, Gaussian, Sobel, etc.) chosen to isolate a specific property (smoothness, edges), while a CNN's convolutional layers *learn* their own kernels from data to isolate whatever features minimize classification error. Applying a classical filter *before* the CNN effectively edits the input distribution the network was pretrained on (ImageNet natural images), which can help if the filter happens to emphasize a feature the network already relies on, or hurt if it destroys information the network needs and can't recover in later layers.

The results bear this out directly: Gaussian smoothing — which preserves color and only mildly reduces high-frequency noise — helped all three models, suggesting the networks' learned features are somewhat robust to (or even benefit from) light denoising. Sobel — which discards color entirely and keeps only edges — hurt all three models severely, showing that these pretrained networks rely heavily on color/texture cues that a hand-crafted edge filter cannot replicate. In short, classical filters are not a substitute for learned features, but they *do* interact with them: a filter that happens to align with what the network already looks for can help, while one that strips away information the network needs will consistently hurt, regardless of architecture.
