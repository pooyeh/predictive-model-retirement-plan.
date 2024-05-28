# predictive_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

# Read data from CSV
df = pd.read_csv('retirement_data.csv')

# Prepare the data
X = df[['age', 'annual_contribution', 'years_to_retirement', 'investment_return_rate']]
y = df['expected_savings']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Save the model
with open('retirement_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Plot the results
plt.scatter(y_test, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Retirement Savings Predictions')
plt.show()
