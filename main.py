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