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

