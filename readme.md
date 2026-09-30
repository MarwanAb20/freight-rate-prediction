# Freight Rate Prediction

Machine Learning regression project for predicting freight rates from shipment, geographic, temporal, and market-related features.

The project implements an end-to-end machine learning pipeline, starting from understanding and validating the data, through feature engineering and chronological model validation, to final model training and prediction generation.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Definition](#2-problem-definition)
- [3. Dataset](#3-dataset)
- [4. Dataset Structure](#4-dataset-structure)
- [5. Data Understanding](#5-data-understanding)
- [6. Data Quality](#6-data-quality)
- [7. Data Cleaning](#7-data-cleaning)
- [8. Feature Engineering](#8-feature-engineering)
- [9. Categorical Features](#9-categorical-features)
- [10. Date Feature Engineering](#10-date-feature-engineering)
- [11. Train/Validation Strategy](#11-trainvalidation-strategy)
- [12. Why Chronological Validation](#12-why-chronological-validation)
- [13. Model Development](#13-model-development)
- [14. Target Transformation](#14-target-transformation)
- [15. Model Comparison](#15-model-comparison)
- [16. Final Model](#16-final-model)
- [17. Final Training](#17-final-training)
- [18. Validation Predictions](#18-validation-predictions)
- [19. December Prediction Task](#19-december-prediction-task)
- [20. Output Validation](#20-output-validation)
- [21. Project Workflow](#21-project-workflow)
- [22. Project Structure](#22-project-structure)
- [23. Installation](#23-installation)
- [24. Running the Project](#24-running-the-project)
- [25. Results](#25-results)
- [26. Key Machine Learning Decisions](#26-key-machine-learning-decisions)
- [27. Technologies](#27-technologies)
- [28. Conclusion](#28-conclusion)

---

# 1. Project Overview

This project focuses on predicting freight rates for shipment loads using supervised machine learning.

The main objective is to learn the relationship between shipment characteristics and the observed freight rate.

The project uses information such as:

- Pickup location
- Delivery location
- Geographic coordinates
- Distance
- Equipment type
- Shipment weight
- Shipment date
- Market index
- Quote signal

The target variable is:

```text
posted_rate
```

The final pipeline uses CatBoost regression and combines numerical, categorical, geographic, route, and temporal features.

---

# 2. Problem Definition

The task is a regression problem.

For each shipment, the model receives information describing the load and its transportation characteristics and predicts the expected freight rate.

Conceptually:

```text
Shipment Information
        │
        ├── Pickup
        ├── Delivery
        ├── Coordinates
        ├── Distance
        ├── Equipment
        ├── Weight
        ├── Date
        ├── Market Index
        └── Quote Signal
                │
                ▼
        Feature Engineering
                │
                ▼
          CatBoost Model
                │
                ▼
        Predicted Freight Rate
```

The objective is to minimize prediction error on unseen future shipment data.

---

# 3. Dataset

The main labeled training dataset contains:

```text
48,000 rows
14 columns
```

The validation dataset contains:

```text
12,000 rows
13 columns
```

The training dataset contains the target variable:

```text
posted_rate
```

The validation dataset does not contain `posted_rate`, because those values must be predicted.

The training data covers:

```text
2025-01-01 → 2025-10-31
```

The validation data covers:

```text
2025-11-01 → 2025-12-31
```

The December prediction task covers:

```text
2025-12-01 → 2025-12-31
```

---

# 4. Dataset Structure

The main training columns are:

```text
load_id
pickup
delivery
pickup_lat
pickup_lon
delivery_lat
delivery_lon
distance
equipment
weight
date
market_index
quote_signal
posted_rate
```

The validation dataset contains the same input features but does not contain:

```text
posted_rate
```

because that is the target to be predicted.

---

# 5. Data Understanding

Before building the model, the dataset was inspected to understand:

- Number of rows and columns
- Feature names
- Numerical and categorical variables
- Target variable
- Missing values
- Duplicate records
- Duplicate shipment IDs
- Date ranges
- Unique pickup locations
- Unique delivery locations
- Equipment categories
- Distribution of the target variable
- Relationships between important numerical variables and the target

The dataset contains:

```text
64 unique pickup locations
64 unique delivery locations
3 equipment categories
```

There are thousands of possible pickup/delivery combinations, making route information potentially useful for the model.

---

# 6. Data Quality

The pipeline performs data quality checks before model training.

The checks include:

```text
Dataset shape
Missing values
Duplicate rows
Duplicate load IDs
Invalid dates
```

The training dataset contains missing values in:

```text
weight          300
market_index    374
```

No duplicate training rows were found.

No duplicate training IDs were found.

No duplicate validation IDs were found.

---

# 7. Data Cleaning

## 7.1 Weight

Some weight values were negative.

Since shipment weight represents a magnitude, negative values were converted to their absolute values:

```python
weight = abs(weight)
```

Missing weight values were not replaced with an arbitrary value.

Instead, the missing values were retained so that CatBoost could handle the missing numerical information.

A separate indicator was also created:

```text
weight_missing
```

This allows the model to distinguish between an observed weight and a missing weight.

---

## 7.2 Numerical Conversion

Numerical columns are explicitly converted to numeric values.

The main numerical variables include:

```text
pickup_lat
pickup_lon
delivery_lat
delivery_lon
distance
weight
market_index
quote_signal
```

Invalid numerical values are converted to missing values instead of causing the entire pipeline to fail.

---

# 8. Feature Engineering

Feature engineering was one of the main parts of the project.

The objective was to provide the model with additional information derived from the original variables.

---

## 8.1 Geographic Features

The dataset contains latitude and longitude for pickup and delivery locations.

Two additional features were created:

```text
lat_diff = delivery_lat - pickup_lat

lon_diff = delivery_lon - pickup_lon
```

These features provide the model with information about the geographic direction and relative position of the shipment.

The original coordinates are also retained.

---

## 8.2 Distance Transformation

The original `distance` feature is retained.

A logarithmic version was also created:

```text
distance_log = log1p(distance)
```

The transformation gives the model an alternative representation of distance and can help model nonlinear relationships.

---

## 8.3 Route Feature

A route feature was created by combining pickup and delivery:

```text
pickup -> delivery
```

For example:

```text
Richmond -> Fort Wayne
```

This allows the model to learn route-specific patterns.

A freight rate can depend not only on the distance but also on the specific origin-destination combination.

---

## 8.4 Missing-Value Indicators

The following missing indicators were created:

```text
weight_missing
market_index_missing
quote_signal_missing
```

These indicators provide the model with explicit information about whether important variables were missing.

---

# 9. Categorical Features

The project uses CatBoost's native categorical feature handling.

The following variables are treated as categorical:

```text
pickup
delivery
equipment
route
```

This is important because locations and equipment types are categories rather than continuous numerical measurements.

For example:

```text
pickup = Richmond
```

should not be interpreted as a numerical value.

Instead, CatBoost handles it as a categorical feature.

Categorical missing values are represented using a dedicated missing category:

```text
__MISSING__
```

---

# 10. Date Feature Engineering

The shipment date contains potentially useful temporal information.

Instead of using the raw date directly, the pipeline extracts several temporal features.

The following features are created:

```text
month
dayofweek
dayofmonth
dayofyear
is_weekend
```

---

## 10.1 Month

The month number is extracted from the shipment date.

```text
month
```

This allows the model to learn monthly patterns.

---

## 10.2 Day of Week

The day of the week is extracted:

```text
dayofweek
```

This can help the model capture weekly freight-rate patterns.

---

## 10.3 Day of Month

The day within the month is extracted:

```text
dayofmonth
```

---

## 10.4 Day of Year

The position of the day within the year is extracted:

```text
dayofyear
```

---

## 10.5 Weekend Indicator

A binary feature identifies weekends:

```text
is_weekend
```

---

## 10.6 Cyclical Date Features

Time-related variables can be cyclical.

For example, the end of a week is close to the beginning of the next week.

To represent this behavior, cyclical encoding is used.

### Month

```text
month_sin
month_cos
```

### Day of Week

```text
dow_sin
dow_cos
```

The cyclical transformations are:

```python
sin(2πx / period)
cos(2πx / period)
```

This allows the model to represent periodic relationships more naturally.

---

# 11. Train/Validation Strategy

A chronological validation strategy was used instead of a random train/test split.

The training data was divided according to time.

### Training period

```text
2025-01-01 → 2025-09-30
```

Number of rows:

```text
43,147
```

### Validation period

```text
2025-10-01 → 2025-10-31
```

Number of rows:

```text
4,853
```

This setup ensures that the model is trained on earlier observations and evaluated on a later period.

---

# 12. Why Chronological Validation

Random splitting was avoided because this is a time-dependent prediction problem.

A random split could allow observations from later periods to appear in the training set while earlier observations appear in validation.

That can make evaluation less representative of how the model will be used in practice.

The chronological split instead simulates:

```text
Past
 │
 ├── Training
 │
 ▼
Future
 │
 └── Validation
```

This provides a more realistic evaluation of the model's ability to generalize to future shipment data.

---

# 13. Model Development

The project compared different target and feature configurations before selecting the final model.

Three configurations were evaluated.

### Model 1

```text
Raw target + core/date features
```

### Model 2

```text
Log target + core/date features
```

### Model 3

```text
Log target + no date features
```

All experiments used the same chronological validation period.

The main selection metric was:

```text
MAE
```

---

# 14. Target Transformation

The target variable `posted_rate` has a wide range and contains relatively high-rate observations.

Two approaches were tested.

## Raw Target

The model directly predicts:

```text
posted_rate
```

## Log Target

The target is transformed using:

```python
log1p(posted_rate)
```

The model learns the transformed target.

After prediction, the transformation is reversed using:

```python
expm1(prediction)
```

Therefore:

```text
posted_rate
      │
      ▼
   log1p()
      │
      ▼
 CatBoost
      │
      ▼
   expm1()
      │
      ▼
predicted_rate
```

The log-target approach produced a lower validation MAE than the raw-target approach in the experiments.

---

# 15. Model Comparison

The three configurations produced the following chronological validation results.

| Model | MAE | RMSE | R² | Best Iteration |
|---|---:|---:|---:|---:|
| Raw target + core/date features | $116.72 | $647.33 | 0.8207 | 200 |
| **Log target + core/date features** | **$108.22** | **$647.21** | **0.8207** | **451** |
| Log target + no date features | $138.80 | $659.74 | 0.8137 | 144 |

The selected configuration was:

```text
Log target + core/date features
```

because it achieved the lowest validation MAE among the tested configurations.

---

# 16. Final Model

The final model uses:

```text
CatBoostRegressor
```

The selected configuration is:

```text
Target:
log1p(posted_rate)

Iterations:
451

Learning rate:
0.04

Depth:
6

L2 regularization:
5

Random seed:
42
```

The model uses both numerical and categorical features.

---

# 17. Why CatBoost

CatBoost was selected because the dataset contains several categorical variables that are potentially important for freight pricing:

```text
pickup
delivery
equipment
route
```

Instead of manually converting categorical values into arbitrary numbers, CatBoost can work with categorical features directly.

This is particularly useful for a dataset containing many different locations and route combinations.

---

# 18. Final Training

After model selection, the selected configuration was retrained using the complete labeled training dataset.

The final training dataset contains:

```text
48,000 rows
```

The final model uses:

```text
27 features
```

The target remains:

```text
log1p(posted_rate)
```

The model is then used to generate predictions for unseen data.

---

# 19. Validation Predictions

The final model predicts all:

```text
12,000
```

validation loads.

The generated output file is:

```text
validation_predictions.csv
```

The output contains exactly:

```text
load_id
predicted_rate
```

The pipeline checks:

- Correct number of predictions
- Correct load IDs
- No duplicate IDs
- No missing predictions
- Positive predictions

The generated predictions in the final run were within approximately:

```text
$205.64 → $6,682.11
```

---

# 20. December Prediction Task

The project also includes a separate December prediction task.

The December input contains:

```text
31 rows
```

representing the dates:

```text
2025-12-01 → 2025-12-31
```

The shipment characteristics are fixed while the date changes.

The final model is used to predict the freight rate for each day.

---

## 20.1 Geographic Information

The December input does not contain all of the geographic coordinate columns required by the model.

To handle this, the pipeline creates a city-to-coordinate lookup from the training data.

The coordinates are then mapped to the December pickup and delivery locations.

This allows December data to pass through the same feature engineering pipeline as the training and validation data.

---

## 20.2 Missing Market Features

The December input does not contain all of the same market-related variables as the training and validation datasets.

The feature engineering function therefore handles optional columns safely.

For example, if:

```text
market_index
```

is not present, the pipeline creates the column as missing rather than failing.

The same logic is used for:

```text
quote_signal
```

This prevents feature engineering errors and keeps the prediction pipeline consistent.

---

## 20.3 Date-Based December Predictions

Because the final model includes date features, predictions are not forced to be identical for every December date.

The final December predictions from the pipeline were approximately:

```text
$834.36 → $854.76
```

across the 31 dates.

The variation comes from the temporal features used by the model.

---

# 21. Output Validation

Before finishing, the pipeline validates the generated outputs.

For the validation predictions it checks:

```text
12,000 rows
Correct load IDs
No duplicate IDs
No missing predictions
Positive predictions
```

For the December predictions it checks:

```text
31 rows
31 unique dates
No missing predictions
Positive predictions
```

This helps prevent malformed output files from being submitted.

---

# 22. Project Workflow

The complete workflow can be summarized as:

```text
                    RAW DATA
                       │
                       ▼
              Data Loading
                       │
                       ▼
             Data Quality Checks
                       │
                       ├── Missing Values
                       ├── Duplicates
                       ├── IDs
                       └── Dates
                       │
                       ▼
              Data Cleaning
                       │
                       ├── Weight Cleaning
                       └── Numeric Conversion
                       │
                       ▼
             Feature Engineering
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
 Geographic        Route Features   Date Features
 Features
       │               │                │
       └───────────────┼────────────────┘
                       │
                       ▼
              Categorical Handling
                       │
                       ▼
          Chronological Validation
                       │
                       ▼
              Model Comparison
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
        Raw        Log + Date    Log no Date
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
               Model Selection
                       │
                       ▼
              Final CatBoost Model
                       │
                       ▼
             Train on All Data
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Validation Prediction   December Prediction
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
                Output Validation
```

---

# 23. Project Structure

The current project structure is:

```text
freight-rate-prediction/
│
├── data/
│   └── raw/
│
├── scorer_results/
│
├── proj_final_code.py
├── score.py
├── validation_predictions.csv
├── requirements.txt
├── readme.md
├── .gitignore
└── freight-rate-ml-assessment.pdf
```

### Main Files

#### `proj_final_code.py`

The main machine learning pipeline.

It performs:

```text
Data loading
Data quality checks
Feature engineering
Chronological validation
Model comparison
Model selection
Final training
Validation prediction
December prediction
Output validation
```

#### `score.py`

The provided scoring script used to validate the generated prediction files.

#### `validation_predictions.csv`

The final predictions for the 12,000 validation loads.

#### `requirements.txt`

Python dependencies required to run the project.

#### `freight-rate-ml-assessment.pdf`

The original assessment/instructions document.

---

# 24. Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# 25. Running the Project

Run the main pipeline:

```bash
python proj_final_code.py
```

The pipeline will automatically:

1. Load the training and validation datasets.
2. Perform data quality checks.
3. Create the chronological training/validation split.
4. Engineer features.
5. Train the candidate models.
6. Compare their validation performance.
7. Select the model with the lowest MAE.
8. Retrain the selected model on all labeled data.
9. Generate validation predictions.
10. Generate December predictions.
11. Validate the generated output files.

---

# 26. Running the Scorer

After running the main pipeline, run:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs.csv
```

The scoring script validates the prediction files according to the assessment requirements.

---

# 27. Results

The chronological validation results from the final pipeline were:

```text
Model:
Log target + core/date features

MAE:
$108.22

RMSE:
$647.21

R²:
0.8207

Best iteration:
451
```

The model comparison showed that using the log-transformed target with date features produced the lowest MAE among the tested configurations.

---

# 28. Key Machine Learning Decisions

Several important decisions were made during development.

## 28.1 Chronological Validation

A chronological split was used instead of a random split to better represent future prediction.

## 28.2 Log-Transformed Target

A log transformation was tested because the target contains a wide range of freight rates.

The log-target model produced lower validation MAE.

## 28.3 Native Categorical Handling

CatBoost was used to handle:

```text
pickup
delivery
equipment
route
```

as categorical features.

## 28.4 Geographic Feature Engineering

Latitude and longitude differences were added to capture geographic relationships.

## 28.5 Route Feature

Pickup and delivery were combined into a route feature to capture route-specific behavior.

## 28.6 Temporal Features

Date-derived features were included because the validation experiments showed that removing date features substantially increased validation MAE.

## 28.7 Missing-Value Indicators

Explicit indicators were added for missing:

```text
weight
market_index
quote_signal
```

so the model can distinguish missing information from observed numerical values.

---

# 29. Limitations and Considerations

The current validation score is based on a single chronological holdout period:

```text
October 2025
```

Therefore, the reported validation metrics should be interpreted as performance on that specific future-like validation period.

The final validation dataset extends beyond the model-selection period, so the final predictions do not have known target values inside this project.

The model also uses a log-transformed target, which can reduce the influence of extreme high-rate observations. This can improve MAE while potentially making extreme-rate predictions more conservative.

---

# 30. Future Improvements

Potential future improvements include:

- Additional time-based features
- More advanced geographic distance calculations
- Route-level historical statistics
- Interaction features
- Additional CatBoost hyperparameter tuning
- Cross-validation strategies designed for time-series data
- Separate analysis of high-rate shipments
- Prediction interval estimation
- Model explainability using SHAP
- Error analysis by route
- Error analysis by equipment type
- Error analysis by distance range
- Monitoring model performance over time

These improvements should be evaluated using a time-aware validation strategy to avoid information leakage.

---

# 31. Technologies

The project uses:

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Matplotlib
- Seaborn
- Jupyter

---

# 32. Conclusion

This project implements a complete machine learning workflow for freight rate prediction.

The development process moved from understanding the raw shipment data to building a validated regression pipeline.

The main stages were:

```text
Data Understanding
        ↓
Data Quality
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Chronological Validation
        ↓
Model Comparison
        ↓
Target Transformation
        ↓
CatBoost Model Selection
        ↓
Final Training
        ↓
Validation Predictions
        ↓
December Predictions
        ↓
Output Validation
```

The final selected approach uses:

```text
CatBoost Regression
+
Log-transformed target
+
Geographic features
+
Distance features
+
Route features
+
Categorical features
+
Market features
+
Temporal features
```

The final chronological validation achieved:

```text
MAE  : $108.22
RMSE : $647.21
R²   : 0.8207
```

The project therefore provides a reproducible end-to-end pipeline for training the model, generating predictions, and validating the required output files.