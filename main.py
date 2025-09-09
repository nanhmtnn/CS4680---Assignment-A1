from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# Load the dataset
data = pd.read_csv('IceCreamData.csv')
print(data.head())

# Prepare features and target
X = data[['Temperature']]
Y = data['Revenue']

# Train the Linear Regression model
model = linear_model.LinearRegression()
model.fit(X, Y)
predictions = model.predict(X)

# Print results for Linear Regression
print("\n--- Linear Regression ---")
print("First 5 predictions vs actual:")
for i in range(5):
    print(f"Temperature: {X.iloc[i,0]:.2f}, Actual Revenue: {Y.iloc[i]:.2f}, Predicted Revenue: {predictions[i]:.2f}")

# Train a different regression model: Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=0)
tree_model.fit(X, Y)
tree_predictions = tree_model.predict(X)

# Print results for Decision Tree Regression
print("\n--- Decision Tree Regression ---")
print("First 5 predictions vs actual:")
for i in range(5):
    print(f"Temperature: {X.iloc[i,0]:.2f}, Actual Revenue: {Y.iloc[i]:.2f}, Predicted Revenue: {tree_predictions[i]:.2f}")

# Sample Output: -----------------------------------------------------

# Temperature     Revenue
# 0    24.566884  534.799028
# 1    26.005191  625.190122
# 2    27.790554  660.632289
# 3    20.595335  487.706960
# 4    11.503498  316.240194

# --- Linear Regression ---
# First 5 predictions vs actual:
# Temperature: 24.57, Actual Revenue: 534.80, Predicted Revenue: 571.63
# Temperature: 26.01, Actual Revenue: 625.19, Predicted Revenue: 602.48
# Temperature: 27.79, Actual Revenue: 660.63, Predicted Revenue: 640.76
# Temperature: 20.60, Actual Revenue: 487.71, Predicted Revenue: 486.47
# Temperature: 11.50, Actual Revenue: 316.24, Predicted Revenue: 291.51

# --- Decision Tree Regression ---
# First 5 predictions vs actual:
# Temperature: 24.57, Actual Revenue: 534.80, Predicted Revenue: 534.80
# Temperature: 26.01, Actual Revenue: 625.19, Predicted Revenue: 625.19
# Temperature: 27.79, Actual Revenue: 660.63, Predicted Revenue: 660.63
# Temperature: 20.60, Actual Revenue: 487.71, Predicted Revenue: 487.71
# Temperature: 11.50, Actual Revenue: 316.24, Predicted Revenue: 316.24