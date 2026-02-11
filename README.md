House Price Prediction Model
This project implements a Machine Learning model to predict house prices using the California Housing Dataset. It explores the complete data science pipeline, including data ingestion, exploratory data analysis (EDA), feature engineering, model training, and hyperparameter tuning.

📊 Project Overview
The goal of this model is to predict the median_house_value for various districts in California based on features like median income, house age, location (latitude/longitude), and ocean proximity.

🛠️ Tech Stack
Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Environment: Google Colab / Jupyter Notebook

🗂️ Dataset
The model uses the California Housing Prices dataset from Kaggle.

Total Records: ~20,640 districts

Key Features:

median_income: Median income for households within a block.

housing_median_age: Median age of a house within a block.

ocean_proximity: Location of the house w.r.t ocean.

total_rooms & total_bedrooms: Count of rooms/bedrooms in the block.

Target Variable: median_house_value.

🚀 Workflow
1. Data Preprocessing
Missing Values: Handled missing data in total_bedrooms by dropping null entries.

Encoding: Categorical features like ocean_proximity were converted into numeric format using One-Hot Encoding.

Scaling: Features were standardized using StandardScaler to improve model convergence.

2. Feature Engineering
Created new informative features to improve model accuracy:

bedroom_ratio: Ratio of bedrooms to total rooms.

household_rooms: Average number of rooms per household.

3. Model Training
Two primary algorithms were evaluated:

Linear Regression: Used as a baseline model.

Random Forest Regressor: An ensemble learning method used to capture non-linear relationships.

4. Optimization
Hyperparameter tuning was performed using GridSearchCV to find the best configuration for the Random Forest model (tuning n_estimators and max_features).

📈 Evaluation
The model's performance was evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) to measure the average error in price prediction.

💻 How to Run
Clone the repository:

Bash
git clone https://github.com/your-username/house-price-prediction.git
Install dependencies:

Bash
pip install pandas numpy scikit-learn matplotlib seaborn
Run the notebook: Open house_price_prediction_model.ipynb in Google Colab or Jupyter and run all cells.
