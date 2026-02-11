# House Price Prediction Model

This project implements a Machine Learning pipeline to predict house prices using the **California Housing Dataset**. It covers everything from data cleaning and exploratory data analysis (EDA) to feature engineering and hyperparameter tuning of a Random Forest model.



## 📊 Project Overview
The objective is to predict the `median_house_value` for various districts in California. This is a regression task where we use geographical, demographic, and housing characteristics to estimate property values.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** * `pandas` & `numpy` (Data Manipulation)
    * `matplotlib` & `seaborn` (Visualization)
    * `scikit-learn` (Machine Learning & Preprocessing)
* **Environment:** Jupyter Notebook / Google Colab

## 🗂️ Dataset
The model uses the **California Housing Prices** dataset.
* **Target Variable:** `median_house_value`
* **Key Features:**
    * `longitude` & `latitude`: Geographical location.
    * `housing_median_age`: Age of the house.
    * `median_income`: Median income of the residents.
    * `ocean_proximity`: Distance/Location relative to the ocean.
    * `total_rooms`, `total_bedrooms`, `population`, `households`.

## 🚀 Pipeline Steps

### 1. Data Cleaning
* Identified and handled missing values in the `total_bedrooms` column.
* Processed categorical data using **One-Hot Encoding** for the `ocean_proximity` feature.

### 2. Exploratory Data Analysis (EDA)
* Visualized correlations using a heatmap to identify which features (like `median_income`) have the strongest impact on house prices.
* Analyzed geographical distribution of prices using latitude and longitude scatter plots.



### 3. Feature Engineering
Improved the model's predictive power by creating derived features:
* `bedroom_ratio`: The ratio of bedrooms to total rooms.
* `household_rooms`: The average number of rooms per household.

### 4. Model Training & Tuning
* **Baseline:** Started with Linear Regression.
* **Advanced Model:** Implemented a **Random Forest Regressor**.
* **Optimization:** Used `GridSearchCV` to find the optimal hyperparameters for the Random Forest model (tuning `n_estimators` and `max_features`).

## 📈 Evaluation
The model's performance was measured using:
* **Root Mean Squared Error (RMSE):** To quantify the average prediction error in dollars.
* **Cross-Validation:** To ensure the model generalizes well to unseen data.

## 💻 How to Use
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/your-username/house-price-prediction.git](https://github.com/your-username/house-price-prediction.git)
