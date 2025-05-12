import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and encoder
model = joblib.load("xgb_model.pkl")        # path to your saved XGBRegressor model
encoder = joblib.load("ordinal_encoder.pkl")  # path to your saved OrdinalEncoder
feature_order = joblib.load("feature_order.pkl")  # saved list of feature names in training order

# Define categorical columns
categorical_cols = ['City', 'Street', 'Statezip(Zip code)']

# App title
st.title("🏠 House Price Prediction App")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter House Details:")

    city = st.selectbox("City", ['Seattle', 'Redmond', 'Bellevue'])  # example values from training set
    street = st.text_input("Street", "Main Street")
    zip_code = st.text_input("Zip code", "98103")
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.0, step=0.5, value=2.0)
    sqft_living = st.number_input("Sqft Living", min_value=100, value=1500)
    sqft_lot = st.number_input("Sqft Lot", min_value=100, value=4000)
    floors = st.number_input("Floors", min_value=1, step=1, value=1)
    condition = st.selectbox("Condition (1-5)", [1, 2, 3, 4, 5], index=3)
    sqft_basement = st.number_input("Sqft Basement", min_value=0, value=0)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    year_renovated = st.number_input("Year Renovated", min_value=0, max_value=2025, value=0)

    submit = st.form_submit_button("Predict Price")

if submit:
    try:
        # Prepare input
        input_data = {
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Sqft_living': sqft_living,
            'Sqft_lot': sqft_lot,
            'Floors': floors,
            'Condition': condition,
            'Sqft_basement': sqft_basement,
            'Year_built': year_built,
            'Year_renovated': year_renovated,
            'City': city,
            'Street': street,
            'Statezip(Zip code)': zip_code
        }

        df = pd.DataFrame([input_data])

        # Encode categorical columns
        df[categorical_cols] = encoder.transform(df[categorical_cols])

        # Reorder columns
        df = df[feature_order]

        # Predict log(price) and transform back
        log_price = model.predict(df)
        price = np.expm1(log_price[0])

        st.success(f"💰 Predicted House Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
