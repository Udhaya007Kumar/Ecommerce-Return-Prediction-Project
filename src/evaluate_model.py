import pandas as pd
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_data

# Load data
df = pd.read_csv(
    'data/ecommerce_return_prediction_dataset.csv'
)

# Preprocess
df = preprocess_data(df)

X = df.drop('Returned_Product', axis=1)

y = df['Returned_Product']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load model
model = joblib.load(
    'models/logistic_model.pkl'
)

# Predict
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title('Confusion Matrix')

plt.savefig(
    'outputs/confusion_matrix.png'
)

# ROC Curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(6, 4))

plt.plot(
    fpr,
    tpr,
    label=f'AUC = {auc:.2f}'
)

plt.plot([0,1], [0,1], '--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()

plt.savefig(
    'outputs/roc_curve.png'
)

# Report
print(classification_report(
    y_test,
    y_pred
))