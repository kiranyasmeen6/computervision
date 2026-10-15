# Landmark Recognition Using SIFT

Lab assignment: recognize famous landmarks from a query photo using the SIFT algorithm (Python + OpenCV). The system compares a new image against a database of reference images, picks the most likely landmark, and rejects images that don't belong to any landmark in the database.

## Landmarks

- Eiffel Tower
- Taj Mahal
- Great Wall of China
- Statue of Liberty
- Colosseum

The Great Pyramid of Giza is used as the "unknown" landmark since it is not in the database.

## How it works

1. **Database:** 16 photos per landmark were downloaded from Wikimedia Commons into separate folders. The first 8 of each landmark are the reference images, and the other 8 are kept aside as test images. 6 pyramid photos are used as the unknown landmark.
2. **SIFT features:** each reference image is converted to grayscale, then keypoints and 128-dimensional descriptors are extracted with `cv2.SIFT_create()`. The descriptors are saved with their landmark labels (40 descriptor sets in total).
3. **Query image:** the unknown image is loaded, converted to grayscale, and its SIFT keypoints and descriptors are extracted.
4. **Matching:** Brute-Force matcher with Euclidean (L2) distance, KNN with k=2, then Lowe's ratio test (ratio 0.75). Reliable matches are counted for each landmark (best reference image of that landmark) and the landmarks are ranked.
5. **Localization:** a homography is estimated with RANSAC (reprojection threshold 5 px). The four corners of the reference image are projected onto the query image and a polygon is drawn around the landmark. If the transformation is not reliable, no polygon is drawn and a message is shown.
6. **Decision:** the largest match count alone is not used. A query is accepted only if all of these are true:

| Condition | Value |
|---|---|
| Reliable matches | at least 15 |
| RANSAC inliers | at least 10 |
| Inliers / reliable matches | at least 0.25 |
| Projected polygon | valid (convex, sensible size) |
| Best landmark vs second best | at least 1.3 times more matches |

Otherwise the image is rejected as "Unknown".

## Database summary

| Landmark | Reference images | Total keypoints | Avg keypoints per image |
|---|---|---|---|
| Eiffel Tower | 8 | 21857 | 2732 |
| Taj Mahal | 8 | 13342 | 1667 |
| Great Wall of China | 8 | 27656 | 3457 |
| Statue of Liberty | 8 | 8497 | 1062 |
| Colosseum | 8 | 22430 | 2803 |

## Files

| File | Description |
|---|---|
| `Landmark_Recognition_SIFT.ipynb` | Full code (Google Colab) |
| `landmark_db/` | Reference and test images, one folder per landmark |
| `unknown_images/` | Images of a landmark that is not in the database |
| `landmark_descriptors.npz` | Saved SIFT descriptors |
| `descriptor_labels.csv` | Landmark label for each saved descriptor set |
| `sift_results.csv` | Results of the performance evaluation |

## How to run

1. Open `Landmark_Recognition_SIFT.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. The first cell downloads the images automatically (no Kaggle login needed). If a landmark ends up with too few images, use the optional upload cell to add your own photos in folders named like `landmark_db/Taj_Mahal/`.
4. The last cells run the tests and save `sift_results.csv`.

## Example outputs

| Query | Best candidate | Reliable matches | RANSAC inliers | Decision |
|---|---|---|---|---|
| Eiffel Tower photo (2447 keypoints) | Taj Mahal | 28 | 6 | Rejected, no polygon drawn |
| Pyramids of Giza (not in database) | Statue of Liberty | 17 | 5 | Rejected, no polygon drawn |

In the Eiffel Tower example the ranking was Taj Mahal 28, Great Wall 23, Statue of Liberty 22, Eiffel Tower 20, Colosseum 18. The correct landmark was only fourth, so the match count alone could not identify it.

## Performance evaluation (Task 6)

Each landmark was tested under 5 conditions (original, rotated 45 degrees, resized to 50%, reduced brightness, different viewpoint), plus 2 images of a landmark not in the database. That is 27 tests in total.

### Full results

| True landmark | Condition | Predicted | Keypoints | Reliable matches | Inliers | Decision | Correct | Time (s) |
|---|---|---|---|---|---|---|---|---|
| Eiffel Tower | Original | Unknown | 2447 | 28 | 6 | Rejected | No | 9.00 |
| Eiffel Tower | Rotated (45 deg) | Unknown | 3211 | 37 | 6 | Rejected | No | 10.98 |
| Eiffel Tower | Resized (50%) | Unknown | 679 | 10 | 4 | Rejected | No | 2.30 |
| Eiffel Tower | Reduced brightness | Unknown | 1546 | 17 | 7 | Rejected | No | 4.56 |
| Eiffel Tower | Different viewpoint | Unknown | 1692 | 19 | 5 | Rejected | No | 5.91 |
| Taj Mahal | Original | Taj Mahal | 1287 | 1099 | 1089 | Recognized | Yes | 4.64 |
| Taj Mahal | Rotated (45 deg) | Taj Mahal | 1739 | 857 | 814 | Recognized | Yes | 5.26 |
| Taj Mahal | Resized (50%) | Taj Mahal | 497 | 358 | 353 | Recognized | Yes | 1.46 |
| Taj Mahal | Reduced brightness | Taj Mahal | 317 | 294 | 293 | Recognized | Yes | 1.21 |
| Taj Mahal | Different viewpoint | Taj Mahal | 1124 | 968 | 960 | Recognized | Yes | 4.84 |
| Great Wall of China | Original | Unknown | 2840 | 31 | 7 | Rejected | No | 10.49 |
| Great Wall of China | Rotated (45 deg) | Unknown | 3927 | 48 | 7 | Rejected | No | 12.68 |
| Great Wall of China | Resized (50%) | Unknown | 858 | 12 | 5 | Rejected | No | 3.96 |
| Great Wall of China | Reduced brightness | Unknown | 610 | 11 | 4 | Rejected | No | 1.89 |
| Great Wall of China | Different viewpoint | Unknown | 2542 | 16 | 6 | Rejected | No | 7.59 |
| Statue of Liberty | Original | Statue of Liberty | 1251 | 60 | 23 | Recognized | Yes | 5.32 |
| Statue of Liberty | Rotated (45 deg) | Unknown | 1710 | 69 | 25 | Rejected | No | 5.22 |
| Statue of Liberty | Resized (50%) | Unknown | 403 | 19 | 9 | Rejected | No | 1.20 |
| Statue of Liberty | Reduced brightness | Statue of Liberty | 457 | 33 | 19 | Recognized | Yes | 1.42 |
| Statue of Liberty | Different viewpoint | Statue of Liberty | 964 | 81 | 66 | Recognized | Yes | 4.58 |
| Colosseum | Original | Unknown | 3512 | 27 | 6 | Rejected | No | 10.51 |
| Colosseum | Rotated (45 deg) | Unknown | 4786 | 57 | 6 | Rejected | No | 17.29 |
| Colosseum | Resized (50%) | Unknown | 1298 | 10 | 5 | Rejected | No | 3.79 |
| Colosseum | Reduced brightness | Unknown | 1418 | 16 | 5 | Rejected | No | 4.19 |
| Colosseum | Different viewpoint | Unknown | 2710 | 49 | 7 | Rejected | No | 10.49 |
| Not in database | Landmark not in database | Unknown | 2262 | 17 | 5 | Rejected | Yes | 7.71 |
| Not in database | Landmark not in database | Unknown | 651 | 19 | 4 | Rejected | Yes | 2.64 |

The "Reliable matches" and "Inliers" values are for the highest-scoring landmark, which is not always the correct one.

### Summary by condition

| Condition | Tests | Correct | Accuracy (%) | Avg time (s) |
|---|---|---|---|---|
| Original | 5 | 2 | 40.0 | 7.99 |
| Rotated (45 deg) | 5 | 1 | 20.0 | 10.29 |
| Resized (50%) | 5 | 1 | 20.0 | 2.54 |
| Reduced brightness | 5 | 2 | 40.0 | 2.65 |
| Different viewpoint | 5 | 2 | 40.0 | 6.68 |
| Landmark not in database | 2 | 2 | 100.0 | 5.18 |

**Overall accuracy: 37.0% (10 of 27 tests correct).**

### Summary by landmark

| Landmark | Correct | Tests |
|---|---|---|
| Taj Mahal | 5 | 5 |
| Statue of Liberty | 3 | 5 |
| Eiffel Tower | 0 | 5 |
| Great Wall of China | 0 | 5 |
| Colosseum | 0 | 5 |
| Not in database | 2 | 2 |

Average processing time was about 6 seconds per query (1.2 s to 17.3 s).

## Discussion

**What worked**
- The system never accepted a wrong landmark. All 8 images it recognized were correct, and both images of the landmark that is not in the database were rejected. The rejection rules (match threshold, RANSAC inliers, inlier ratio, margin over the second-best landmark) kept false recognitions at zero.
- Taj Mahal was recognized in all 5 tests, and Statue of Liberty in 3 of 5.
- Reduced brightness did not hurt recognition where the landmark could be matched at all (Statue of Liberty was still recognized with 33 matches and 19 inliers).

**What did not work**
- Eiffel Tower, Great Wall of China and Colosseum were rejected in all 15 of their tests. Their reliable matches were only 10 to 57, and the RANSAC inliers were only 4 to 7, below the minimum of 10. The match counts for different landmarks overlap (for the Eiffel Tower original, Taj Mahal scored 28 and the Eiffel Tower 20), so many of those matches are chance matches and not true correspondences.
- The test photos are different photographs from the reference photos, taken from other angles, distances and lighting, so there were few true matches. Large structures with a lot of changing perspective are hard for SIFT, which is only partly invariant to 3D viewpoint change.
- Rotating the image to 45 degrees gave more keypoints (for example 3211 instead of 2447 for the Eiffel Tower) because the image canvas gets larger, but the extra matches were mostly not reliable.
- Resizing to 50% reduced the keypoints a lot (for example 1251 to 403 for the Statue of Liberty), which gave fewer matches. The Statue of Liberty resized test had 9 inliers, just below the minimum of 10.
- The Statue of Liberty rotated test had 69 matches and 25 inliers but was still rejected, so it was stopped by one of the other checks (polygon validity or the margin over the second-best landmark).

**About the Taj Mahal result**
The Taj Mahal tests have very high numbers (294 to 1099 matches, with about 99% of them inliers). That is much higher than for any other landmark, and it suggests the Taj Mahal test images are near-duplicates of images that are also in the reference set (the same photo uploaded more than once to Wikimedia Commons, for example). So the 100% for Taj Mahal is probably higher than the system would score on truly new photos, and the real overall accuracy is probably lower than 37%.

## Limitations

- Few true matches when the viewpoint or distance changes a lot, which is the main reason three landmarks were never recognized.
- The decision thresholds were chosen by hand and are fairly strict. They avoid false recognitions but also reject many correct images.
- 8 reference images per landmark is small. More reference images from different angles and distances would give more chances for a good match.
- The photos come from a general image search, so some may be near-duplicates across the reference and test sets, or may show only part of the landmark.
- Every query is matched against all 40 reference images, which takes up to 17 seconds on Colab for images with many keypoints.

## Possible improvements

- Add more reference images per landmark with different angles, distances and lighting.
- Check for and remove duplicate images between the reference and test sets.
- Try lowering the minimum inlier threshold slightly and check that false recognitions stay at zero.
- Use RootSIFT or a FLANN-based matcher to improve matching quality and speed.
- Limit the number of keypoints per image to reduce the processing time.

## Requirements

- Python 3, OpenCV (with SIFT, version 4.4 or newer), NumPy, pandas, matplotlib, requests (all available in Google Colab)

## References

- D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," International Journal of Computer Vision, 2004.
- D. Vaithiyanathan and M. Manigandan, "Real-time-based Object Recognition using SIFT algorithm," ICEEICT, 2023.
