import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
import pickle

warnings.filterwarnings("ignore")

# Load the dataset
file_path = 'airline_changed.csv'
df = pd.read_csv(file_path)
print(df.head())

# Check the column names
print("Column names:\n", df.columns)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Drop rows with missing values in the 'Arrival Delay' column
df = df.dropna(subset=['Arrival Delay'])

# Check for missing values after dropping
missing_values_after = df.isnull().sum()
print("Missing values after dropping rows:\n", missing_values_after)

# Prepare data
X = df.drop(columns=['Satisfaction'])
y = df['Satisfaction']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Define parameters and train the RandomForest model
params_rf = {'n_estimators': 100, 'max_depth': 24}
model = RandomForestClassifier(**params_rf)
model.fit(X_train, y_train)

# Save the model
pickle.dump(model, open('airline.pkl', 'wb'))
