import pandas as pd
import joblib

# Load model
model = joblib.load(
    'models/logistic_model.pkl'
)

scaler = joblib.load(
    'models/scaler.pkl'
)

encoders = joblib.load(
    'models/encoder.pkl'
)


def predict_return(data_dict):

    df = pd.DataFrame([data_dict])

    # Encode
    for col, encoder in encoders.items():

        df[col] = encoder.transform(
            df[col]
        )

    # Scale
    df = scaler.transform(df)

    # Predict
    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability