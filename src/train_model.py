import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

from data_preprocessing import preprocess_data

# Load dataset
df = pd.read_csv(
    'data/ecommerce_return_prediction_dataset.csv'
)

# Preprocess
df = preprocess_data(df)

# Features and target
X = df.drop('Returned_Product', axis=1)

y = df['Returned_Product']

# Handle imbalance
smote = SMOTE(random_state=42)

X, y = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Hyperparameter tuning
params = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

model = LogisticRegression()

grid = GridSearchCV(
    model,
    params,
    cv=5,
    scoring='f1'
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print('Best Parameters:')
print(grid.best_params_)

# Save model
joblib.dump(
    best_model,
    'models/logistic_model.pkl'
)

print('Model Saved Successfully')