import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
df = pd.read_csv('C:/Users/Sumit/Downloads/rainfall in india 1901-2015.csv')

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']] = imputer.fit_transform(df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

# Encode the state names as integers
state_to_int = {state: i for i, state in enumerate(df['SUBDIVISION'].unique())}
df['state_encoded'] = df['SUBDIVISION'].map(state_to_int)


# Create the input features and target variables
X = df[['state_encoded', 'YEAR']]
y = df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Calculate accuracy (not recommended for regression problems)
y_test_rounded = y_test.round()
y_pred_rounded = pd.DataFrame(y_pred, columns=y_test.columns).round()
accuracy = accuracy_score(y_test_rounded.values.ravel(), y_pred_rounded.values.ravel())
print(f"Accuracy: {accuracy:.2f}")

# Example of making a prediction
state = 'ANDAMAN & NICOBAR ISLANDS'
year = 1901
state_encoded = state_to_int[state]
prediction = model.predict([[state_encoded, year]])[0]

print(f"Predicted rainfall amounts for {state} in {year}:")
for month, amount in zip(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], prediction):
    print(f"{month}: {amount:.2f}")

#model
model_filename = 'rainfall.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model has been saved to {model_filename}")

# Save the state-to-integer mapping
state_mapping_filename = 'state_to_int.pkl'
with open(state_mapping_filename, 'wb') as file:
    pickle.dump(state_to_int, file)
print(f"State-to-integer mapping has been saved to {state_mapping_filename}")