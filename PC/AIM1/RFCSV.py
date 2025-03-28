import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Folder containing CSV files
csv_folder = "LCG"

# Read all CSV files into a single DataFrame with a random sample of frac% of the rows
dataframes = []
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        # Use on_bad_lines='skip' to skip problematic lines
        df = pd.read_csv(os.path.join(csv_folder, file), on_bad_lines='skip')
        # Randomly sample frac% of the rows
        df = df.sample(frac=0.15, random_state=7)
        dataframes.append(df)
df = pd.concat(dataframes, ignore_index=True)

# Drop rows where 'Endgame' is null
df = df.dropna(subset=['Endgame'])

# Convert problematic columns to numeric, coercing errors to NaNs
df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
df.iloc[:, 3] = pd.to_numeric(df.iloc[:, 3], errors='coerce')

# Drop rows with missing values in the entire DataFrame
df = df.dropna()

# Convert 'Date' into separate day, month, and year columns
df[['year', 'month', 'day']] = df['Date'].str.split('.', expand=True).astype(int)
df.drop(columns=['Date'], inplace=True)

# Label encoding
def encode_column(column):
    le = LabelEncoder()
    return le.fit_transform(column)

df['Endgame'] = encode_column(df['Endgame'])
df['Result'] = encode_column(df['Result'])
df['Event'] = encode_column(df['Event'])
df['Opening'] = encode_column(df['Opening'])
df['ECO'] = encode_column(df['ECO'])
df['TimeControl'] = encode_column(df['TimeControl'])
df['Termination'] = encode_column(df['Termination'])

# Feature scaling (using pandas for scaling)
scaled_features = ['WhiteElo', 'BlackElo', 'day', 'month', 'year']
for feature in scaled_features:
    df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

# Features and labels
X = df.drop('Endgame', axis=1)  # Features
y = df['Endgame']  # Target variable

# Filter out rare classes
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 480000].index

X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

print("Class distribution after filtering rare classes:", Counter(y))

# Balance classes using SMOTE
smote = SMOTE(random_state=7, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("Class distribution after balancing:", Counter(y_balanced))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=7
)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,  # Number of trees in the forest
    criterion='gini',  # Splitting criterion
    max_depth=20,    # No maximum depth (nodes expanded until pure)
    min_samples_split=2,  # Minimum samples required to split a node
    min_samples_leaf=1,   # Minimum samples required at a leaf node
    max_features='sqrt',  # Number of features to consider at each split
    random_state=7,      # Random seed for reproducibility
    n_jobs=-1,           # Use all available processors
    verbose=1            # Show progress during fitting
)

rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Random Forest Accuracy: {accuracy:.2f}')
print(f'Random Forest F1 Score (Weighted): {f1:.2f}')
print(classification_report(y_test, y_pred))

# Plot feature importance
importances = rf_model.feature_importances_
indices = importances.argsort()

plt.figure(figsize=(16, 6))
plt.title('Random Forest Feature Importances')
plt.bar(range(len(indices)), importances[indices], align='center')
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()

# Save the plot as an EPS file
plt.savefig('rf_feature_importances.eps', format='eps')

# Show the plot
plt.show()