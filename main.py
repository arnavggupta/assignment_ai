import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("new-thyroid.data", header=None)

# Display the first few rows of the dataset to understand its structure
print(data.head())


X = data.iloc[:, 1:6]  
Y = data.iloc[:, 0]   

# Handle categorical variables by encoding them
# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Apply label encoding to each categorical feature in X
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = label_encoder.fit_transform(X[column])

# Encode the target variable if it's categorical
if Y.dtype == 'object':
    Y = label_encoder.fit_transform(Y)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
model = GaussianNB()

# Train the model
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')