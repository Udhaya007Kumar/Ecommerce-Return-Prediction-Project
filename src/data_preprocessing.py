import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def preprocess_data(df):

    # Drop unnecessary columns
    df.drop(columns=['Order_ID'], inplace=True)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df['Order_Amount'].fillna(df['Order_Amount'].median(), inplace=True)

    df['Product_Rating'].fillna(
        df['Product_Rating'].median(),
        inplace=True
    )

    df['Product_Category'].fillna(
        df['Product_Category'].mode()[0],
        inplace=True
    )

    # Handle outliers
    q1 = df['Order_Amount'].quantile(0.25)
    q3 = df['Order_Amount'].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df = df[
        (df['Order_Amount'] >= lower) &
        (df['Order_Amount'] <= upper)
    ]

    # Encoding
    categorical_cols = [
        'Product_Category',
        'Payment_Method',
        'Customer_Type',
        'City'
    ]

    encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoders[col] = le

    # Scaling
    scaler = StandardScaler()

    feature_cols = df.drop(
        'Returned_Product',
        axis=1
    ).columns

    df[feature_cols] = scaler.fit_transform(
        df[feature_cols]
    )

    # Save preprocessors
    joblib.dump(
        scaler,
        'models/scaler.pkl'
    )

    joblib.dump(
        encoders,
        'models/encoder.pkl'
    )

    return df