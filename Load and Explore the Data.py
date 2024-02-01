import pandas as pd

# Load the dataset
url = "https://drive.google.com/uc?id=1KWE3J0uU_sFIJnZ74Id3FDBcejELI7FD"
df = pd.read_csv(url)

# Explore the data
print(df.head())
print(df.info())
print(df.describe())

# Separate features (X) and target variable (Y)
X = df.iloc[:, 1:]  # Features (X1 to X6)
Y = df['Y']  # Target variable

# Check for missing values
print(X.isnull().sum())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, Y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)

# Analyze feature importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
sorted_importance = feature_importance.sort_values(ascending=False)
print("Feature Importance:\n", sorted_importance)

# Plot feature importance
import matplotlib.pyplot as plt
sorted_importance.plot(kind='barh')
plt.title('Feature Importance')
plt.show()

from sklearn.feature_selection import SelectFromModel

# Use a feature selection approach
sfm = SelectFromModel(model)
sfm.fit(X_train, Y_train)

# Selected features
selected_features = X.columns[sfm.get_support()]
print("Selected Features:", selected_features)
