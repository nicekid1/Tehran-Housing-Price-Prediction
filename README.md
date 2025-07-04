#  Tehran Housing Price Prediction

A data science project aimed at building a robust predictive model to estimate housing prices in **Tehran** based on real estate listing data.

##  Project Objective

The primary goal is to develop an accurate and generalizable model that can estimate housing prices using structured features such as **Area**, **Room Count**, **Location**, and **Facilities** (e.g., Parking, Elevator, Warehouse). 

This model can serve as a decision-support tool for:
- **Buyers and sellers** looking to assess fair market prices
- **Urban planners** analyzing real estate dynamics
- **Developers and investors** making data-driven decisions

---

##  Workflow Overview

The project follows a complete machine learning pipeline:

### 1. Data Loading & Exploration
- Loaded housing data into pandas DataFrame
- Inspected data types, null values, and basic statistics

### 2. Exploratory Data Analysis (EDA)
- Explored distributions, relationships, and outliers
- Analyzed price distribution by room count, facilities, and area
- Examined correlation between price and other numerical features

### 3. Missing Value Handling
- Imputed missing `Area` values based on average area per room
- Dropped rows with missing `Address` (low count)

### 4. Feature Engineering
Created additional features to boost model performance:
- `Facilities_sum`: total number of available facilities
- `Room_per_Area`: room density
- `Neighborhood_house_count`: listing frequency by address

### 5. Data Preprocessing
- Label encoded `Address` into `AddressEncode`
- Applied log transformations to `Price` and `Area` for normalization
- Standardized features where required

### 6. Model Training and Evaluation

#### Traditional ML Models:
- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest, KNN
- Gradient Boosting: **XGBoost, LightGBM, CatBoost**
- Stacking Ensemble (XGBoost + Decision Tree + Linear Regression)

#### Deep Learning:
- Fully Connected ANN with BatchNorm and EarlyStopping using TensorFlow/Keras
- Compared performance with ML models

---

## Model Comparison

| Model             | MAE (IRR)          | RMSE (IRR)         | R² Score | MAE (USD)   | RMSE (USD)  |
|-------------------|--------------------|--------------------|----------|-------------|-------------|
| XGBoost           | 1,472,786,064      | 4,652,751,348      | 0.730    | 16,256  | 51,355  |
| Stacking Ensemble  | 1,638,821,150      | 4,578,545,931      | 0.738    | 18,089  | 50,536 |
| ANN (Deep Learning)| 1,886,608,949      | 4,693,735,608      | 0.725    | 20,823  | 51,807  |

> 💲 USD values calculated based on an exchange rate of 1 USD = 90,000 IRR

---

## Key Insights

- Tree-based models like **XGBoost** provided strong results and are suitable for this regression task.
- **Stacking** further improved the R² score, indicating better generalization.
- **Deep Learning** models performed competitively, but boosting-based models remain superior for this dataset.
- **Feature engineering** and **log transformations** were crucial for boosting accuracy.

---


