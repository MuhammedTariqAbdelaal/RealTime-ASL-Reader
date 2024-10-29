#Import libraries
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load the processed data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Create a machine learning pipeline with data scaling and a classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('classifier', RandomForestClassifier())  # Use a Random Forest classifier
])

# Define a set of hyperparameters to tune the model
param_grid = {
    'classifier__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4],    # Minimum samples required at a leaf node
    'classifier__bootstrap': [True, False]        # Whether to use bootstrapped samples
}

# Set up a grid search to find the best combination of hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Train the model using the best parameters found
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_

# Test the model
y_predict = best_model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# show classification report
print("\nClassification Report:")
print(classification_report(y_test, y_predict))

# show the average accuracy
cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring='accuracy')
print(f'Average cross-validation accuracy: {np.mean(cv_scores) * 100:.2f}%')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': best_model}, f)

print("Model training and saving completed successfully.")
