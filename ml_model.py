
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
# Step 1: Generate synthetic data
# Creating a classification dataset with 1000 samples and 10 features
X, y = make_classification(
    n_samples=1000,  # Number of samples
    n_features=4,   # Total number of features
    n_informative=2, # Number of informative features
    n_redundant=2,   # Number of redundant features
    random_state=42  # Seed for reproducibility
)

# Step 2: Split the data into training and testing sets
# 70% training data, 30% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Step 4: Define the hyperparameter grid for tuning
param_grid = {
    'criterion': ['gini', 'entropy'],          # Splitting criteria
    'max_depth': [40, 50, 60, None],            # Maximum depth of the tree
    'min_samples_split': [10, 20, 40],          # Minimum samples required to split an internal node
    'min_samples_leaf': [5, 6, 10]             # Minimum samples required to be at a leaf node
}

# Step 5: Perform Grid Search with Cross-Validation
# GridSearchCV to find the best hyperparameters for the model
grid_search = GridSearchCV(
    estimator=clf,          # Base estimator
    param_grid=param_grid,  # Hyperparameter grid
    cv=5,                   # 3-fold cross-validation
    scoring='accuracy',     # Scoring metric
    verbose=1,              # Verbosity level for tracking progress
    n_jobs=-1               # Use all available cores for parallel processing
)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Step 6: Best parameters and model evaluation
# Extract the best model from GridSearchCV
best_clf = grid_search.best_estimator_

# Print the best parameters
print("Best Parameters:", grid_search.best_params_)

# Predictions on the test set
y_pred = best_clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detailed classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix for better insight into predictions
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# joblib.dump(best_clf, 'model.pkl')
best_clf = grid_search.best_estimator_

# Optionally, you can print or log the model
print("Best Model: ", best_clf)

from sklearn.model_selection import cross_val_score

print("Feature Importances:", best_clf.feature_importances_)
test_inputs = np.array([
    [-2.47, 0.81, 0.46, 0.23],
    [-0.42, -0.19, 0.83, -0.36],
    [2.09, -0.40, -1.03, 0.15]
])
predictions = best_clf.predict(test_inputs)
print("Predictions:", predictions)

