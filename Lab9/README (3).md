# Comparing 6 Regression Algorithms on California Housing

**Name:** Kiran Yasmeen
**University:** COMSATS University Islamabad, Wah Campus
**Program:** BS Artificial Intelligence

## Results

All six models used the same features, the same target (`MedHouseVal`) and the same 80:20 train/test split on the California Housing dataset (20,640 rows, 8 features).

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **XGBoost** | 0.3107 | 0.2176 | 0.4665 | 0.8339 |
| Random Forest | 0.3275 | 0.2554 | 0.5053 | 0.8051 |
| Neural Network | 0.3505 | 0.2743 | 0.5237 | 0.7907 |
| Decision Tree | 0.4482 | 0.4220 | 0.6497 | 0.6779 |
| KNN | 0.4462 | 0.4324 | 0.6576 | 0.6700 |
| Linear Regression | 0.5332 | 0.5559 | 0.7456 | 0.5758 |

## Strengths and Limitations

| Algorithm | Strengths | Limitations |
|---|---|---|
| Linear Regression | Fast, simple, easy to interpret | Captures only linear patterns; sensitive to outliers |
| Decision Tree | Easy to understand; handles non-linear data; no scaling needed | Overfits easily; unstable |
| Random Forest | Accurate; reduces overfitting; robust to noise | Slower; less interpretable |
| Neural Network | Learns complex patterns | Needs scaling and tuning; black box; needs more data |
| XGBoost | Very accurate; regularization; fast training | Many hyperparameters; can overfit if not tuned |
| KNN | Simple; no training phase | Slow on large data; needs scaling; hurt by irrelevant features |

## Conclusion

**XGBoost performed the best.** It had the lowest MAE, MSE and RMSE and the highest R² (0.8339), meaning it explains about 83% of the variation in house prices. Random Forest came second and the Neural Network third. Linear Regression did the worst because house prices do not follow a simple straight-line relationship with the features.

Ensemble methods (XGBoost and Random Forest) performed best on this dataset, so XGBoost is the most suitable model for house price prediction in this experiment.
