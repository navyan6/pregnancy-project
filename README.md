AI-Based Prediction of High-Risk Pregnancies

Machine learning models for early identification of high-risk pregnancies using demographic, clinical, and laboratory data, with emphasis on interpretability and deployment feasibility.

**Overview**:

	This project frames pregnancy risk prediction as supervised learning under two settings:

Binary classification: high-risk vs normal pregnancy (demographic + clinical data)

Multiclass classification: low / medium / high risk (laboratory measurements)

The objective is early risk stratification using routinely collected patient data.

**Data**:

Public, de-identified maternal health datasets (~1k samples each)

Features include age, chronic conditions, blood pressure, blood glucose, heart rate, and body temperature

Data split: 70% train / 10% validation / 20% test

**Models**:

Regularized GLM (L1/LASSO): linear baseline with feature selection

Gradient Boosted Trees (LightGBM): nonlinear modeling for tabular data

Multilayer Perceptron (MLP): feedforward neural network to model higher-order feature interactions

Hyperparameters were tuned using grid search and Optuna (TPE sampler).

**Evaluation**:

Models were evaluated using:

ROC-AUC, Average Precision

Precision, Recall, F1

One-vs-rest / one-vs-one AUC for multiclass classification

Tree-based models performed best overall, with strong performance on lab-based risk prediction.

**Interpretability & Risk Scoring**:

Feature importance derived from GLM coefficients and GBM gain/split statistics

Model outputs mapped to continuous risk scores via the inverse logit function

Decision paths visualized to support clinical interpretability

**Deployment**:

A mobile app (Hera) was built using Kivy, allowing clinicians to input patient data and receive a normalized pregnancy risk score.

**Tech Stack**:

Python · scikit-learn · LightGBM · Optuna · NumPy · Pandas · Kivy
