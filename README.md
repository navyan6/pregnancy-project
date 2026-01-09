# AI-Based Prediction of High-Risk Pregnancies

Machine learning models for early identification of high-risk pregnancies using demographic, clinical, and laboratory data, with an emphasis on interpretability and deployment feasibility.

## Overview

This project frames pregnancy risk prediction as a supervised learning problem under two settings:

- **Binary classification**  
  High-risk vs. normal pregnancy (demographic + clinical data)

- **Multiclass classification**  
  Low / medium / high risk (laboratory measurements)

The objective is early risk stratification using routinely collected patient data.

## Data

- Public, de-identified maternal health datasets (~1,000 samples each)
- Features include:
  - Age  
  - Chronic conditions  
  - Blood pressure  
  - Blood glucose  
  - Heart rate  
  - Body temperature  
- Data split:
  - 70% training
  - 10% validation
  - 20% test

## Models

- **Regularized GLM (L1 / LASSO)**  
  Linear baseline with feature selection

- **Gradient-Boosted Trees (LightGBM)**  
  Nonlinear modeling for tabular data

- **Multilayer Perceptron (MLP)**  
  Feedforward neural network to model higher-order feature interactions

Hyperparameters were tuned using grid search and Optuna (TPE sampler).

## Evaluation

Metrics used:
- ROC-AUC  
- Average Precision  
- Precision  
- Recall  
- F1-score  

Tree-based models performed best overall, with strong performance on lab-based risk prediction.

## Interpretability & Risk Scoring

- Feature importance derived from GLM coefficients and GBM gain/split statistics
- Model outputs mapped to continuous risk scores via the inverse logit function
- Decision paths visualized to support clinical interpretability

## Deployment

A mobile app (**Hera**) was built using **Kivy**, allowing clinicians to input patient data and receive a normalized pregnancy risk score.

## Tech Stack

Python · scikit-learn · LightGBM · Optuna · NumPy · Pandas · Kivy
