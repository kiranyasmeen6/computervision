# Skin Lesion Boundary Detection Using Canny Edge Detection — Results & Questions

## Required Results Table

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
|---|---|---|---|---|
| Image 1 | Gaussian | Canny (50-100) | 44.0 | 31.3 |
| Image 2 | Gaussian | Canny (50-100) | 26.0 | 25.7 |
| Image 3 | Gaussian | Canny (50-100) | 62.0 | 40.6 |
| Image 4 | Gaussian | Canny (50-100) | 0.0 | 0.0 |
| Image 5 | Gaussian | Canny (50-100) | 1386.0 | 983.7 |

**Note on Image 4:** no contour was detected at all (area = 0, perimeter = 0) — the edge map for this image didn't produce a closed boundary that `cv2.findContours` could pick up, even after the morphological closing step. This is a real limitation of the method, not a bug — see Question 5 below.

**Note on Image 5:** area/perimeter are far larger than the other four images (1386 px vs 26–62 px). This likely means the detected "lesion boundary" actually traced a much larger region of the image (possibly the image border, a large shadow/vignette, or a genuinely large lesion) rather than a tight lesion outline — worth a manual visual check against `task5_boundaries.png` before trusting this number.

## Final Comparison Table

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
|---|---|---|---|---|
| Original + Sobel | Poor | Noisy/Dense | Partial/fragmented | Poor |
| Original + Canny | Poor | Balanced | Partial/fragmented | Fair |
| Average + Sobel | Poor | Noisy/Dense | Closed contour found | Fair |
| Average + Canny | Good | Sparse | No usable contour | Fair |
| Gaussian + Sobel | Poor | Noisy/Dense | Closed contour found | Fair |
| Gaussian + Canny | Poor | Sparse | Partial/fragmented | Poor |
| Median + Sobel | Poor | Noisy/Dense | Closed contour found | Fair |
| Median + Canny | Moderate | Sparse | Partial/fragmented | Poor |

**Key pattern:** Sobel consistently found a closed contour (Average/Gaussian/Median + Sobel all got "Closed contour found"), but at the cost of a noisy/dense edge map and poor noise handling. Canny tended toward sparse edge maps that were more fragmented or, in one case (Average + Canny), didn't produce a usable contour at all. No single combination scored well on all three criteria at once — this is reflected in every row capping out at "Fair" or worse for Overall Performance.

## Questions to Answer

**1. Why is Gaussian filtering applied before Canny detection?**

Canny edge detection works by computing intensity gradients, and gradients are very sensitive to noise — a single noisy pixel can create a large, spurious gradient that gets mistaken for a real edge. Applying a Gaussian filter first smooths out that noise while preserving the overall intensity transitions at true object boundaries, so the gradient computation that follows responds to genuine lesion-boundary transitions rather than pixel-level noise. This matches what the Final Comparison Table shows: "Original + Canny" (no pre-filtering) still only reached "Fair" with a "Partial/fragmented" boundary, while filtering was needed across the board to get any kind of consistent contour at all.

**2. How did the three Canny threshold settings affect the result?**

| Setting | Avg Edge Density (%) | Avg Largest-Contour Area (% of image) |
|---|---|---|
| 50–100 | 0.64 | 0.03 |
| 100–200 | 0.11 | 0.02 |
| 150–250 | 0.01 | 0.01 |

Edge density dropped sharply as the thresholds increased — from 0.64% at 50–100 down to just 0.01% at 150–250, roughly a 64x decrease. This is the expected direction (higher thresholds = more selective = fewer detected edges), but the magnitude is striking: even the lowest threshold setting (50–100) only detected edges on 0.64% of pixels, and the two higher settings detected almost nothing (0.11% and 0.01%). This suggests these lesion images have fairly low contrast at their boundaries, so even a "low" threshold pair here is still fairly strict relative to the actual gradient strengths present.

**3. Which threshold produced the best lesion boundary?**

**50–100** was selected, and by a clear margin — it had both the highest edge density (0.64% vs 0.11% and 0.01%) and the largest average contour area (0.03% vs 0.02% and 0.01%) of the three settings. Since the other two settings were detecting almost no edges at all, 50–100 was really the only setting of the three that found enough edge pixels to produce usable contours in the first place.

**4. Why are edges useful for detecting skin lesions?**

A skin lesion is typically visually distinguished from surrounding healthy skin by a relatively sharp transition in color/intensity at its border — the lesion is darker/differently pigmented than the skin around it. Edge detection is specifically designed to find exactly these kinds of sharp local intensity transitions, which makes it a natural tool for locating where the lesion's boundary is, independent of the lesion's internal color or texture pattern.

**5. What problems did you observe in detecting the lesion boundary?**

Several concrete problems showed up in this run:
- **No contour found at all for Image 4** (area/perimeter both 0) — the edge map simply didn't produce a closed shape `cv2.findContours` could use, even with morphological closing applied first.
- **Very low edge density overall** — even the best threshold setting (50–100) only marked 0.64% of pixels as edges on average, meaning Canny was only picking up a small fraction of what might be true boundary; the 100–200 and 150–250 settings detected almost nothing (0.11% and 0.01%), confirming these lesion images have fairly low-contrast borders for a fixed-threshold approach.
- **Sobel vs. Canny trade-off** — Sobel consistently produced a closed contour (useful for actually measuring area/perimeter), but its edge maps were rated "Noisy/Dense" and had "Poor" noise handling. Canny's edge maps were cleaner ("Balanced" or "Sparse") but more often fragmented or, in one case, failed to produce a usable contour at all.
- **A likely oversized boundary for Image 5** (area 1386 px vs 26–62 px for the other four) — suggests the detected contour may have traced something other than just the lesion itself.
- **No combination in the Final Comparison Table scored well on Noise Handling, Edge Quality, and Boundary Detection simultaneously** — every row capped at "Fair" or "Poor" overall, showing a genuine trade-off between these three properties with this classical approach.

**6. How could your method be improved?**

A few directions, some of them directly motivated by the problems above: (a) use an **adaptive/per-image threshold** (e.g. Otsu's method to pick Canny's thresholds automatically from each image's own intensity histogram) instead of one fixed setting for every image — given how different the edge densities were even across the 3 tested settings, a fixed threshold clearly isn't well-matched to every image; (b) apply **color-based segmentation** (e.g. in HSV or LAB color space) alongside or instead of pure edge-based detection, since lesion boundaries are often more reliably separated by color difference than by a pure intensity gradient — this would likely also fix the Image 4 case where no edge-based contour was found at all; (c) use a **learned segmentation model** (e.g. a U-Net trained specifically for lesion segmentation) which can learn to ignore hair/noise artifacts and generalize much better than a fixed classical pipeline, instead of the Sobel-vs-Canny trade-off seen here where neither method scored well on all three criteria; (d) add a **hair-removal preprocessing step** (a common dermoscopy preprocessing technique) before edge detection, since body hair is a frequent source of false edges in this kind of image and could explain some of the "Noisy/Dense" ratings for the Sobel combinations.
