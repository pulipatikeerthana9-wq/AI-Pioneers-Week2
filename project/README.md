# Week 2: Supervised Machine Learning Models

This project compares regression and classification baselines using datasets bundled with scikit-learn. It is designed to be reproducible and to evaluate models on held-out test data.

## Run

```bash
python -m pip install -r requirements.txt
python run_experiments.py
```

The script writes `results/regression_metrics.csv`, `results/classification_metrics.csv`, and `results/experiment_metadata.json`.

## Experiments

- **Regression:** Diabetes dataset (442 observations, 10 input features); Linear Regression, standardized 5-neighbor KNN, and Random Forest. 80/20 holdout split (seed 42). MAE and RMSE are errors in the dataset target's units; lower is better. R² measures explained variance on the test split; higher is better.
- **Classification:** Breast Cancer Wisconsin dataset (569 observations, 30 features); Logistic Regression, Decision Tree, standardized 5-neighbor KNN, and Random Forest. Stratified 80/20 split (seed 42). Malignant is explicitly encoded as the positive class. Precision, recall, F1, and ROC-AUC are therefore reported for malignant cases. Higher is generally better, though precision/recall involve trade-offs.

Scaling is inside pipelines for the scale-sensitive models and is fitted only on training data. The random forest and decision tree use no scaling. The test set is not used for training or preprocessing fit.

## Limitations

These are benchmark experiments on small built-in datasets, using one holdout split and simple default parameters. Results are dataset- and split-specific, are not clinical guidance, and do not establish generalization to new institutions or populations. No hyperparameter tuning or cross-validation is performed.
