# Lab 03 Report – Edge Detection Techniques and Their Impact on Classification Performance

## 1. Introduction

For this lab I implemented the main edge detection methods (Sobel, Prewitt, Laplacian, LoG and Canny) and tested how they behave when noise is added to the image. I also compared how well a model classifies skin lesions when it's given raw images vs filtered images vs edge-only images. Basically the goal was to see if giving the model "just the edges" helps or hurts, since a lot of the diagnostic info for skin lesions (like color) isn't really an edge at all.

## 2. Methodology

I used the same 4 classes from Lab 1/Lab 2 (nv, mel, bkl, bcc), 250 images each.

For edge detection I implemented:
- Sobel (Gx, Gy and the magnitude)
- Prewitt
- Laplacian
- LoG (Laplacian of Gaussian)
- Canny

For noise I added Gaussian noise and salt-and-pepper noise, then tried cleaning them up with a Gaussian filter and a Median filter (Gaussian filter paired with Gaussian noise, Median with salt-and-pepper, since that's the "correct" pairing).

For Canny I tried 3 different threshold pairs plus one with a bigger 5x5 kernel to see what changes.

For classification I used 3 classical ML models (SVM, Random Forest, KNN) trained on features pulled from a pretrained ResNet18, plus 2 CNNs trained end-to-end (ResNet18 and VGG16 — these were the top models from Lab 1). All 5 models were run on 3 versions of the dataset: Raw, Filtered (Gaussian, since that was the best filter from Lab 2), and Edge (best Canny setup from Task 3). Same train/val/test split and same number of epochs (5) for everyone so it's a fair comparison.

## 3. Experimental Setup

- Dataset: HAM10000, 4 classes, 1000 images
- Tools: PyTorch/torchvision for the CNNs, scikit-learn for SVM/RF/KNN, OpenCV for all the filtering and edge detection stuff
- Ran on Google Colab with the free T4 GPU
- 5 epochs for both CNNs (didn't have time/compute for more)

## 4. Results

### Table 1 – Effect of Noise and Preprocessing on Edge Detection

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Density (%) | Noise Sensitivity (%) | Observations |
|---|---|---|---|---|---|---|
| Sobel | Original | None | None | 99.39 | 0.00 | baseline |
| Sobel | Noisy | Gaussian | None | 99.96 | 95.73 | very noisy, lots of false edges |
| Sobel | Noisy | Salt & Pepper | None | 96.46 | 82.95 | also bad but less than gaussian |
| Sobel | Noisy | Gaussian | Gaussian Filter | 99.84 | 92.16 | barely improved honestly |
| Sobel | Noisy | Salt & Pepper | Median Filter | 95.25 | 55.31 | big improvement here |
| Prewitt | Original | None | None | 99.49 | 0.00 | baseline only |
| Laplacian | Original | None | None | 93.62 | 0.00 | baseline only |
| LoG | Noisy | Gaussian | Gaussian Filter | 87.08 | 93.43 | still really high even after smoothing |
| Canny | Original | None | Built-in smoothing | 0.01 | 0.00 | barely detects anything at default settings |
| Canny | Noisy | Gaussian | Gaussian Filter | 0.00 | 0.14 | basically same as before, both near 0 |
| Canny | Noisy | Salt & Pepper | Median Filter | 0.00 | 0.14 | same story |

Quick note on the Edge Density numbers — for Sobel/Prewitt/Laplacian/LoG they're basically all sitting at 90%+ which looked weird to me at first. I think this is because I'm normalizing the gradient image with min-max scaling before counting nonzero pixels, so almost every pixel ends up above 0 even if the "real" edges are much weaker than the background. So density isn't that useful for these 4 detectors — the Noise Sensitivity column (which compares each result to its own clean baseline using SSIM) is the one that actually tells you something.

Things I noticed:
- Sobel with Gaussian noise had the single highest sensitivity score in the whole table (95.73%). Makes sense, it's just a derivative with no smoothing built in.
- Gaussian filtering barely helped Sobel against Gaussian noise (only went from 95.73 to 92.16). I expected more of a drop honestly.
- Median filtering did a lot better job against salt-and-pepper (82.95 down to 55.31). That's a much bigger improvement.
- Canny basically found nothing on this particular image at the default 50/150 thresholds — I didn't expect that, but it explains why I needed Task 3 to tune the thresholds in the first place.

### Table 2 – Canny Parameter Analysis

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | # Detected Edges | Observation |
|---|---|---|---|---|---|---|
| Canny-1 | 30 | 100 | 3x3 | sparse | 245 | 0.37% pixels |
| Canny-2 | 50 | 150 | 3x3 | sparse | 8 | 0.01% pixels |
| Canny-3 | 100 | 200 | 3x3 | sparse | 0 | 0% pixels, detects nothing |
| Canny-4 | 50 | 150 | 5x5 | dense/noisy | 18765 | 28.63% pixels |

Best config I picked: **Canny-1 (30, 100, 3x3)**. It's still pretty sparse but out of the 4 options it was the closest to a reasonable amount of edges. Canny-3 literally found zero edges which isn't useful at all, and Canny-4 way over-detected (went from 8 edges to almost 19,000 just by switching the kernel size, which surprised me — I thought the thresholds would matter more than the kernel size but apparently not).

### Table 3 – Cross-Lab Classification Performance Comparison

| Model / Classifier | Acc. Raw (%) | Acc. Filtered (%) | Acc. Edge (%) | Precision (%) | Recall (%) | F1 (%) | Training Time (s) | Inference Time (ms) |
|---|---|---|---|---|---|---|---|---|
| SVM | 60.67 | 59.33 | 42.00 | 60.73 | 60.67 | 59.79 | 0.74 | 0.337 |
| Random Forest | 64.00 | 62.67 | 42.67 | 63.84 | 64.00 | 63.04 | 2.59 | 0.110 |
| KNN | 50.67 | 48.67 | 44.00 | 50.11 | 50.67 | 49.99 | 0.00 | 0.463 |
| CNN Model 1 (ResNet18) | **67.33** | 66.67 | 45.33 | 68.05 | 67.33 | 66.54 | 43.40 | 12.653 |
| CNN Model 2 (VGG16) | 60.67 | 60.00 | 45.33 | 69.51 | 60.67 | 56.03 | 67.37 | 13.131 |

(Precision/Recall/F1/times are from each model's best dataset, which for every single model turned out to be Raw.)

Best model overall was ResNet18 on Raw images, 67.33% accuracy.

### Task 6 – Confusion Matrices & Bar Chart

For ResNet18 (best model) comparing Raw/Filtered/Edge:

| Dataset | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Raw | 67.33 | 68.05 | 67.33 | 66.54 |
| Filtered | 66.67 | 66.12 | 66.67 | 66.18 |
| Edge | 45.33 | 45.72 | 45.33 | 45.44 |

Raw and Filtered are basically the same, but Edge drops off a cliff — about 21-22 points lower on every metric. See `task6_confusion_matrices.png` for the actual confusion matrices.

## 5. Discussion

**Q1: Which edge detector was most sensitive to noise?**

Sobel, pretty clearly. It hit 95.73% noise sensitivity when I added Gaussian noise, which was the highest number in the whole table. This is because Sobel is just a plain derivative operator, it doesn't smooth the image at all before computing gradients, so any noisy pixel directly becomes a fake edge. LoG was also bad (93.43%) even though it has a Gaussian blur step built in — I guess the Laplacian part afterward still amplifies whatever noise is left over after the blur. I didn't test Prewitt/Laplacian under noise in my run (only did the baseline for those), so I can't say for sure how they'd compare, but since they're built on similar math to Sobel I'd guess they'd behave about the same.

**Q2: How did Gaussian and Median filtering affect edge quality?**

Not equally well. Gaussian filtering only dropped Sobel's noise sensitivity from 95.73 to 92.16 against Gaussian noise — like a 3-4 point improvement, not much. Median filtering did way better against salt-and-pepper noise, dropping sensitivity from 82.95 down to 55.31, almost a 28 point improvement. This kind of matches what you'd expect from the theory — median filtering is supposed to be really good at removing those random black/white outlier pixels that salt-and-pepper noise creates, because the median just throws out extreme values instead of averaging them in like a Gaussian filter would.

**Q3: How did changing Canny's thresholds affect the results?**

Pretty straightforward — higher thresholds = fewer edges detected. Went from 245 edges (30/100) down to 8 edges (50/150) down to literally 0 edges (100/200). What I didn't expect was that changing the kernel size mattered so much more than the thresholds. Same exact thresholds (50/150) but switching from a 3x3 to 5x5 kernel took it from 8 edges to almost 19,000. I think this is because the bigger kernel produces much bigger raw gradient values before they even get compared to the threshold.

**Q4: Did edge images improve or hurt accuracy compared to raw?**

Hurt it, for every single model, no exceptions:
- SVM: 60.67 → 42.00 (down 18.67)
- Random Forest: 64.00 → 42.67 (down 21.33)
- KNN: 50.67 → 44.00 (down 6.67, smallest drop)
- ResNet18: 67.33 → 45.33 (down 22.00, biggest drop)
- VGG16: 60.67 → 45.33 (down 15.34)

The two CNNs dropped the most, which makes sense since they were pretrained on regular color ImageNet photos — their early layers are expecting color and texture info that just isn't there in an edge map. KNN dropped the least, but that's probably because it wasn't doing great even on raw images to begin with (only 50.67%), so it didn't have as much to lose.

**Q5: What information gets lost with edge maps?**

Basically everything except the outline. You lose color (which for skin lesions is honestly a huge deal — pigmentation is one of the main things dermatologists look at), texture, and any shading/gradient info that's inside the lesion instead of at its border. The accuracy drops in Q4 are basically direct proof of how much useful info was in that stuff.

**Q6: Why let a CNN learn its own edge features instead of giving it edge maps?**

Because a CNN can learn a bunch of different edge-like filters on its own from the data, in different orientations and scales, and it doesn't have to throw away color/texture to do it — those get combined together in later layers. Handing it a pre-made edge map forces it to work with just one fixed definition of "edge" and nothing else, which the results here show is clearly worse.

**Q7: Which representation worked best overall?**

Raw won for every single model I tested — no exceptions. Filtered was really close behind (within about 1-2 points for all of them). Edge was clearly the worst, by a lot (15-22 points behind Raw for the CNNs). So overall: Raw > Filtered > Edge, with Raw and Filtered being pretty close and Edge being way behind. Best single result was ResNet18 on Raw at 67.33%.

## 6. Conclusion

Raw images gave the best classification results across the board, filtered images (Gaussian) were basically just as good, and edge-only images were clearly the worst option, losing anywhere from about 7 to 22 accuracy points depending on the model. On the edge detection side, Sobel was the most noise-sensitive detector I tested, median filtering worked a lot better against salt-and-pepper noise than Gaussian filtering did against Gaussian noise, and Canny's kernel size mattered way more than I expected compared to its thresholds. Overall I'd say classical edge detection is a useful tool for visualizing lesion boundaries, but it's not a good replacement for giving a CNN the full image — you lose too much color/texture information that the model actually needs.

## 7. References

- Gonzalez, R. C., & Woods, R. E. Digital Image Processing.
- Canny, J. (1986). A Computational Approach to Edge Detection. IEEE TPAMI.
- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. Scientific Data.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition.
- Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.

---

## Viva Questions

1. **What is an edge in an image?** A spot where the pixel intensity changes suddenly — usually where one object/region ends and another begins.

2. **Difference between first-order and second-order edge detection?** First-order (Sobel, Prewitt) uses the first derivative/gradient and looks for high magnitude. Second-order (Laplacian, LoG) uses the second derivative and looks for zero-crossings instead. Second-order methods are more sensitive to noise since you're differentiating twice.

3. **Difference between Sobel Gx and Gy?** Gx picks up horizontal changes in intensity (so it finds vertical edges), Gy picks up vertical changes (finds horizontal edges). Combine both to get the full gradient magnitude.

4. **Why is Laplacian more sensitive to noise?** It's a second derivative, and taking derivatives amplifies high-frequency stuff — noise is high-frequency, so it gets amplified twice as much.

5. **Why smooth before edge detection?** To get rid of noise first so the derivative doesn't pick up random pixel fluctuations as fake edges.

6. **Main advantage of Canny?** It does smoothing + gradient + thinning (non-max suppression) + a two-threshold system all together, so you get cleaner, thinner, more connected edges with fewer false positives than the simpler methods.

7. **What are Canny's low/high thresholds?** Above the high threshold = definitely an edge. Below the low threshold = definitely not an edge. In between = only counted if it's connected to a strong edge pixel.

8. **Gaussian noise vs Salt-and-Pepper noise?** Gaussian adds a small random value to every pixel (grainy look everywhere). Salt-and-pepper randomly turns some pixels fully white or fully black (sparse extreme spots).

9. **Why does median filtering work well on salt-and-pepper noise?** Because it picks the middle value in a neighborhood, so the extreme 0/255 outlier pixels just get ignored instead of dragging the average up or down like they would with a mean/Gaussian filter.

10. **Why can edge detection hurt classification?** Because you're throwing away color, texture and shading info, and most pretrained models actually rely on that info a lot — my results showed accuracy dropping 7-22 points when switching from raw to edge images.

11. **Can a CNN learn edge features on its own?** Yes, the early conv layers of CNNs trained on regular images pretty much always end up learning edge/gradient-like filters automatically without being told to.

12. **Why might raw images beat edge-only images?** Because raw images have everything (color, texture, shape) and the model can pick whatever's useful, while edge images only give it one narrow slice of information (shape/boundaries) and nothing else.