import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

st.set_page_config(
    page_title="Customer Prediction App",
    page_icon="📊"
)

# Sidebar for switching between Classification and Regression using radio buttons
st.sidebar.title("Choose Prediction Type")
option = st.sidebar.radio(
    "Select:",
    ["Classification (Churn Prediction)", "Regression (Salary Prediction)"]
)

if option == "Classification (Churn Prediction)":
    # Load the classification model
    model = load_model('churn_model.h5')

    # Load the scaler
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    # Load the encoders
    geo_encoder = pickle.load(open('onehotencoder_geo.pkl', 'rb'))
    gen_encoder = pickle.load(open('label_encoder_gender.pkl', 'rb'))

    # Streamlit App for Classification
    st.title('Customer Churn Prediction 🔍')

    st.write('Enter the details of the customer to predict if they will churn (leave) or not')
    # User Input
    geo = st.selectbox('Geography', geo_encoder.categories_[0])
    gender = st.selectbox('Gender', gen_encoder.classes_)
    age = st.slider('Age', 18, 100)
    tenure = st.slider('Tenure', 0, 10)
    balance = st.number_input('Balance')
    salary = st.number_input('Estimated Salary')
    products = st.slider('Number of Products', 1, 4)
    credit = st.number_input('Credit Score')
    has_card = st.selectbox('Has Credit Card', [0, 1])
    is_active = st.selectbox('Is Active Member', [0, 1])

    input_data = pd.DataFrame({
        'CreditScore': [credit],
        'Gender': [gen_encoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [salary],
    })

    # Encode Geography
    geo_encoded = geo_encoder.transform([[geo]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

    # Concatenate the data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    pred_proba = prediction[0][0]

    if pred_proba > 0.5:
        st.write('The customer is likely to churn')
    else:
        st.write('The customer is not likely to churn')

    st.write(f'Probability of Churn: {pred_proba:.2f}')

elif option == "Regression (Salary Prediction)":
    # Load the regression model
    model = load_model('regression_model.h5')

    # Load the encoders
    geo_encoder = pickle.load(open('onehotencoder_geo_reg.pkl', 'rb'))
    gen_encoder = pickle.load(open('label_encoder_gender_reg.pkl', 'rb'))
    scaler = pickle.load(open('scaler_reg.pkl', 'rb'))

    # Streamlit App for Regression
    st.title('Estimated Salary Prediction 💰')

    st.write('Enter the customer details to predict their estimated salary')
    # User Input
    geo = st.selectbox('Geography', geo_encoder.categories_[0])
    gender = st.selectbox('Gender', gen_encoder.classes_)
    age = st.slider('Age', 18, 100)
    tenure = st.slider('Tenure', 0, 10)
    balance = st.number_input('Balance')
    products = st.slider('Number of Products', 1, 4)
    credit = st.number_input('Credit Score')
    has_card = st.selectbox('Has Credit Card', [0, 1])
    is_active = st.selectbox('Is Active Member', [0, 1])
    exited = st.selectbox('Exited (0 = Not exited, 1 = Exited)', [0, 1])  # Input for Exited

    # Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [credit],
        'Gender': [gen_encoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [has_card],
        'IsActiveMember': [is_active],
        'Exited': [exited],  # Include Exited as a feature
    })

    # Encode Geography
    geo_encoded = geo_encoder.transform([[geo]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))

    # Concatenate the data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Ensure feature consistency (reorder columns to match training data)
    expected_features = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    input_data = input_data[expected_features]  # Reorder columns to match training data

    # Scale the data
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    st.write(f'Predicted Estimated Salary: {prediction[0][0]:.2f}')
