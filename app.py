import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊"
)
# Load the model
model = load_model('churn_model.h5')

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the encoders
geo_encoder = pickle.load(open('onehotencoder_geo.pkl', 'rb'))
gen_encoder = pickle.load(open('label_encoder_gender.pkl', 'rb'))

# Streamlit App

st.title('Customer Churn Prediction 🔍')

st.write('Enter the details of the customer to predict if they will churn (leave) or not')
# User Input

geo=st.selectbox('Geography', geo_encoder.categories_[0])
gender=st.selectbox('Gender',gen_encoder.classes_)
age=st.slider('Age', 18, 100)
tenure=st.slider('Tenure', 0, 10)
balance=st.number_input('Balance')
salary=st.number_input('Estimated Salary')
products=st.slider('Number of Products', 1, 4)
credit=st.number_input('Credit Score')
has_card=st.selectbox('Has Credit Card', [0,1])
is_active=st.selectbox('Is Active Member', [0,1])

input_data=pd.DataFrame({
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

