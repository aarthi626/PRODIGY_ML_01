# PRODIGY_ML_01
the task is about implementing a linear regression model for the provided dataset for the purpose of predicting the house prices based upon sq.feet and no.of bedrooms and bathrooms
House Price Prediction using Linear Regression 🏡
A simple Python project that predicts house prices based on square footage, number of bedrooms, and bathrooms using a linear regression model.

Description
This project demonstrates a fundamental machine learning workflow. It takes a dataset of housing information, trains a linear regression model on specific features (area, bedrooms, bathrooms), and evaluates its performance. The final output shows a comparison between the actual house prices and the prices predicted by the model, formatted as clean, whole numbers.

Dataset
The model is trained on the Housing.csv dataset. This file contains various attributes of houses, but for this specific model, we only use the following columns:

price: The target variable (actual price of the house).

area: The independent variable (total square footage).

bedrooms: The independent variable (number of bedrooms).

bathrooms: The independent variable (number of bathrooms).

Getting Started
Follow these instructions to get the project running on your local machine.

# Prerequisites
You need to have Python installed. The project uses the following Python libraries:

1.pandas 


2.scikit-learn

# Pythoncode


import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_squared_error, r2_score

# --- 1. Load the Dataset ---
# Ensure 'Housing.csv' is in the same directory as this script.
df = pd.read_csv('Housing.csv')

# --- 2. Select Features and Target ---
features = ['area', 'bedrooms', 'bathrooms']
X = df[features]
y = df['price']

# --- 3. Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Create and Train the Linear Regression Model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 5. Make Predictions ---
y_pred = model.predict(X_test)

# --- 6. Evaluate the Model ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"Features Used: {features}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}\n")

# --- 7. Create and Display the Breakdown Table ---
results_display_df = X_test.copy()
results_display_df['Actual Price'] = y_test
results_display_df['Predicted Price'] = y_pred

# Convert 'Predicted Price' to whole numbers (integers)
results_display_df['Predicted Price'] = results_display_df['Predicted Price'].astype(int)

results_display_df = results_display_df.rename(columns={
    'area': 'Sq. Feet',
    'bedrooms': 'Bedrooms',
    'bathrooms': 'Bathrooms'
})

final_display_df = results_display_df[['Sq. Feet', 'Bedrooms', 'Bathrooms', 'Actual Price', 'Predicted Price']]

print("--- Price Prediction Breakdown ---")
print(final_display_df.head())
#Output
Running the script will print the model's performance metrics followed by a table comparing the actual and predicted prices for a sample of the test data.

--- Model Performance ---
Features Used: ['area', 'bedrooms', 'bathrooms']
Mean Squared Error: 2750040479309.0522
R-squared: 0.4559299118872445

--- Price Prediction Breakdown ---
     Sq. Feet  Bedrooms  Bathrooms  Actual Price  Predicted Price
316      5900         4          2       4060000          6383168
77       6500         3          2       6650000          6230250
360      4040         2          1       3710000          3597885
90       5000         3          1       6440000          4289731
493      3960         3          1       2800000          3930446
