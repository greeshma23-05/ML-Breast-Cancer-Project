# Breast Cancer Classification with Robust and Interpretable Machine Learning

This project is an end-to-end machine learning study on classifying malignant vs. benign breast tumors using the Breast Cancer Wisconsin dataset. I built this project to go beyond basic model training and focus on **reproducibility, evaluation rigor, robustness, calibration, and interpretability**, which are critical for real-world and research-oriented ML systems.

Rather than optimizing a single metric, the project emphasizes understanding **when models fail**, **how confident their predictions are**, and **how performance changes under realistic distribution shifts**.

---

## Project Goals

The main goals of this project are to:
- Build a **clean, reproducible ML pipeline** from raw data to evaluation
- Establish **strong and honest baselines** using proper cross-validation
- Analyze **probability calibration** and confidence reliability
- Stress-test the model under **distribution shifts and noise**
- Use **SHAP interpretability** to understand global drivers and individual predictions
- Practice ML engineering practices beyond notebook-only workflows

---

## Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Task:** Binary classification (malignant = 1, benign = 0)
- **Size:** 569 samples, 30 numerical features
- **Target distribution:** ~37% malignant, ~63% benign

Raw and processed data are intentionally **not committed** to the repository.  
The dataset can be regenerated locally using the provided scripts.

---

## Project Structure

ML-Breast-Cancer-Project/
├── src/
│ ├── data/ # Data loading and preprocessing
│ ├── models/ # Training, CV, calibration, SHAP, stress tests
│ └── evaluation/ # Metrics, calibration, robustness, plotting
├── tests/ # Unit tests for evaluation utilities
├── reports/
│ ├── figures/ # Generated plots (not tracked in git)
│ └── *.json # Local evaluation artifacts (not tracked)
├── README.md
└── .gitignore

yaml
Copy code

---

## Modeling and Evaluation

### Baseline Models
- Logistic Regression
- Random Forest
- Gradient Boosting (HistGradientBoostingClassifier)

### Metrics Used
- ROC-AUC
- PR-AUC
- Sensitivity (Recall for malignant cases)
- Specificity
- Expected Calibration Error (ECE)

### Cross-Validation
- Stratified 5-fold cross-validation
- Out-of-fold (OOF) probability predictions
- Single evaluation on OOF predictions to avoid optimistic bias

---

## Probability Calibration

I evaluated probability calibration using:
- **Uncalibrated logistic regression**
- **Platt scaling (sigmoid)**
- **Isotonic regression**

Calibration quality was measured using **Expected Calibration Error (ECE)** and reliability diagrams.

**Key observation:**  
Isotonic calibration produced the lowest ECE, improving probability reliability at the cost of a small reduction in ranking performance (ROC-AUC), highlighting the tradeoff between discrimination and calibration.

---

## Distribution Shift Stress Testing

To evaluate robustness, I tested the trained model under multiple distribution shifts:

1. **Gaussian noise injection** (simulated measurement noise)
2. **Systematic feature drift** (bias in a key feature)
3. **Subpopulation slice evaluation** (harder or extreme cases)

**Findings:**
- Performance degrades sharply under increasing noise
- The model is relatively robust to systematic feature drift
- Errors are concentrated near decision boundaries rather than extreme cases

These experiments highlight that **noise sensitivity**, not drift, is the primary failure mode for this model.

---

## Uncertainty & Reject Option

I implemented a confidence-based reject option:
- Confidence defined as `max(p, 1 − p)`
- Low-confidence predictions are abstained
- Evaluated the tradeoff between **coverage** and **error rate**

This analysis shows how abstention can significantly reduce false negatives in high-stakes settings at the cost of reduced coverage.

---

## Interpretability (SHAP)

To understand model behavior, I used **SHAP (TreeExplainer)** on a Random Forest model:

- **Global explanations**
  - Feature importance bar plot
  - Beeswarm plot showing feature effects across samples
- **Local explanations**
  - SHAP waterfall plots for:
    - A true positive (correct malignant prediction)
    - A false negative (missed malignant case)

The false negative explanation provides insight into which feature values pushed the model toward a benign decision, helping connect interpretability with safety and threshold analysis.

---

## Reproducibility

To run the project locally:

```bash
# create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# generate processed dataset
python -m src.data.make_dataset

# run baseline models
python -m src.models.train_baseline

# run cross-validation
python -m src.models.train_cv

# run calibration analysis
python -m src.models.calibrate_cv

# run robustness tests
python -m src.models.run_shift_tests

# run SHAP explanations
python -m src.models.shap_explain
