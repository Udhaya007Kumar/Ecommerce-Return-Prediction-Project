# Ecommerce Return Prediction System

A professional Machine Learning web application that predicts whether a customer is likely to return a purchased product based on customer behavior, order details, and transaction history.

---

# Project Overview

Ecommerce companies face major losses due to product returns. This project helps businesses analyze customer purchase patterns and predict return probability using Machine Learning algorithms.

The system uses customer and order-related features such as:

- Customer Age
- Order Amount
- Delivery Days
- Previous Returns
- Product Rating
- Product Category
- Payment Method
- Customer Type
- City

The application predicts:

- Whether the product is likely to be returned
- Return probability percentage
- Risk level analysis

---

# Key Features

## Machine Learning Features

- Logistic Regression Classification
- Feature Scaling using StandardScaler
- Label Encoding for categorical variables
- SMOTE for class imbalance handling
- Hyperparameter tuning using GridSearchCV
- Probability-based prediction
- ROC-AUC evaluation

---

## Web Application Features

- Interactive Streamlit Dashboard
- Real-time customer input
- Dynamic return prediction
- Plotly interactive charts
- Professional UI design
- Risk-level analysis

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-learn | Machine Learning |
| Streamlit | Web Application |
| Plotly | Interactive Visualization |
| Matplotlib | Graph Plotting |
| Seaborn | Statistical Visualization |
| Joblib | Model Serialization |
| Imbalanced-learn | SMOTE Oversampling |

---

# Project Structure

```txt
Ecommerce_Return_Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── ecommerce_return_prediction_dataset.csv
│
├── models/
│   ├── logistic_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
│
├── requirements.txt
└── README.md

Dataset Collection
        ↓
Data Cleaning
        ↓
Handling Missing Values
        ↓
Outlier Removal
        ↓
Feature Encoding
        ↓
Feature Scaling
        ↓
SMOTE Balancing
        ↓
Train-Test Split
        ↓
Model Training
        ↓
Hyperparameter Tuning
        ↓
Model Evaluation
        ↓
Prediction Deployment


| Feature          | Description             |
| ---------------- | ----------------------- |
| Customer_Age     | Age of customer         |
| Order_Amount     | Total purchase amount   |
| Delivery_Days    | Delivery duration       |
| Previous_Returns | Previous return count   |
| Product_Rating   | Customer product rating |
| Product_Category | Product category        |
| Payment_Method   | Payment option          |
| Customer_Type    | New / Regular / Premium |
| City             | Customer city           |
| Returned_Product | Target variable         |

Model Used
Logistic Regression

Logistic Regression is used for binary classification:

Returned Product → 1
Not Returned → 0

The model predicts probability values between:

0 → Low Return Risk
1 → High Return Risk


Performance Evaluation
The model is evaluated using:

Accuracy Score
Precision
Recall
F1-Score
ROC-AUC Score
Confusion Matrix



Run Streamlit Application
streamlit run app/app.py
Output Files
File	Purpose
logistic_model.pkl	Saved ML model
scaler.pkl	Saved scaler
encoder.pkl	Saved encoders
confusion_matrix.png	Evaluation graph
roc_curve.png	ROC visualization