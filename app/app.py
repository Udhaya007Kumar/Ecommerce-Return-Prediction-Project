import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

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

# Title
st.title('Ecommerce Return Prediction')

st.sidebar.header('Customer Input')

# Inputs
customer_age = st.sidebar.slider(
    'Customer Age',
    18,
    70,
    30
)

order_amount = st.sidebar.number_input(
    'Order Amount',
    500,
    100000,
    5000
)

delivery_days = st.sidebar.slider(
    'Delivery Days',
    1,
    15,
    5
)

previous_returns = st.sidebar.slider(
    'Previous Returns',
    0,
    10,
    1
)

product_rating = st.sidebar.slider(
    'Product Rating',
    1,
    5,
    3
)

product_category = st.sidebar.selectbox(
    'Product Category',
    [
        'Electronics',
        'Fashion',
        'Home',
        'Beauty',
        'Sports'
    ]
)

payment_method = st.sidebar.selectbox(
    'Payment Method',
    [
        'UPI',
        'Credit Card',
        'Debit Card',
        'Cash on Delivery'
    ]
)

customer_type = st.sidebar.selectbox(
    'Customer Type',
    [
        'New',
        'Regular',
        'Premium'
    ]
)

city = st.sidebar.selectbox(
    'City',
    [
        'Chennai',
        'Bangalore',
        'Mumbai',
        'Delhi',
        'Hyderabad'
    ]
)

# Create dataframe
input_df = pd.DataFrame({

    'Customer_Age': [customer_age],

    'Order_Amount': [order_amount],

    'Delivery_Days': [delivery_days],

    'Previous_Returns': [previous_returns],

    'Product_Rating': [product_rating],

    'Product_Category': [product_category],

    'Payment_Method': [payment_method],

    'Customer_Type': [customer_type],

    'City': [city]
})

# Encode
categorical_cols = [
    'Product_Category',
    'Payment_Method',
    'Customer_Type',
    'City'
]

for col in categorical_cols:

    input_df[col] = encoders[col].transform(
        input_df[col]
    )

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button('Predict Return'):

    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader('Prediction Result')

    # HIGH RISK
    if probability >= 0.75:

        st.error('⚠ High Return Risk')

        st.metric(
            label='Return Probability',
            value=f'{probability*100:.1f}%'
        )

        st.warning(
            'Customer has a very high chance of returning the product.'
        )

        st.info(
            'Recommendation: Offer faster delivery or discounts.'
        )

    # MEDIUM RISK
    elif probability >= 0.40:

        st.warning('🟠 Medium Return Risk')

        st.metric(
            label='Return Probability',
            value=f'{probability*100:.1f}%'
        )

        st.info(
            'Customer may return the product.'
        )

    # LOW RISK
    else:

        st.success('✅ Low Return Risk')

        st.metric(
            label='Return Probability',
            value=f'{probability*100:.1f}%'
        )

        st.success(
            'Customer is likely to keep the product.'
        )

    # Probability Chart
    st.subheader('Prediction Analysis')

    chart_df = pd.DataFrame({
        'Category': [
            'Return Chance',
            'Retention Chance'
        ],
        'Value': [
            probability,
            1 - probability
        ]
    })

    fig = px.pie(
        chart_df,
        names='Category',
        values='Value',
        hole=0.5,
        title='Return Prediction Probability'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
# Footer
