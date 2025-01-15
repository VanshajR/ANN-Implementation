# Customer Churn Prediction using Artificial Neural Networks (ANN)
This project implements an Artificial Neural Network (ANN) to predict customer churn using a dataset of customer data. It also does prediction of estimated salary using regression The implementation includes data preprocessing, model training, and deployment of a prediction app using Streamlit.
Check it out here: 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ann-prac-vanshajr.streamlit.app)
## Table of Contents
- [Customer Churn Prediction using Artificial Neural Networks (ANN)](#customer-churn-prediction-using-artificial-neural-networks-ann)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Setup and Installation](#setup-and-installation)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Making Predictions](#making-predictions)
    - [Using the Streamlit App](#using-the-streamlit-app)
  - [Model Details](#model-details)
  - [Results](#results)
  - [Additional Comments](#additional-comments)

## Overview
Customer churn prediction is crucial for businesses to retain customers and reduce revenue loss. This project trains an ANN model on customer data to predict whether a customer is likely to leave (churn). The application includes:

- Preprocessing categorical and numerical data.
- Training a Sequential ANN model.
- Providing an interactive prediction app using Streamlit.

## Features
- Data Preprocessing:

  - Encodes categorical features using Label Encoding and One-Hot Encoding.
  - Scales numerical features using StandardScaler.

- Model Training:

  - Utilizes a TensorFlow Sequential ANN model with two hidden layers.
  - Incorporates callbacks for early stopping and TensorBoard logging.

- Deployment:

  - Interactive prediction app developed using Streamlit.
  - Allows real-time predictions based on user inputs.

## Project Structure
```bash
.
├── salary_regression.ipynb      # Preprocessing and training regression model
├── model_creation.ipynb         # Preprocessing and training classification model
├── predict.ipynb                # Notebook for testing predictions (classification)
├── app.py                       # Streamlit app for user interaction
├── churn_model.h5               # Trained ANN model for classification
├── regression_model.h5          # Trained ANN model for regression
├── scaler.pkl                   # Saved scaler for feature scaling (classification)
├── scaler_reg.pkl               # Saved scaler for feature scaling (regression)
├── onehotencoder_geo.pkl        # One-hot encoder for geography (classification)
├── onehotencoder_geo_reg.pkl    # One-hot encoder for geography (regression)
├── label_encoder_gender.pkl     # Label encoder for gender (classification)
├── label_encoder_gender_reg.pkl # Label encoder for gender (regression)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

```

## Setup and Installation
1. Clone the repository:
```bash
git clone https://github.com/VanshajR/ANN-Implementation.git
cd ANN-Implementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage
### Training the Model
1. Open `model_creation.ipynb` and `salary_regression.ipynb` in Jupyter Notebook.

2. Run the notebook cells sequentially to preprocess data, train the ANN, and save the model and encoders.

### Making Predictions
1. Open `predict.ipynb` in Jupyter Notebook.

2. Provide new customer data in the input_data dictionary.

3. Run the notebook cells to preprocess and predict customer churn.

### Using the Streamlit App
1. Launch the app using the `streamlit run app.py` command.

2. Fill in customer details using the interactive form.

3. View the churn prediction and estimated salary with respective probabilities.

## Model Details
- Architecture:

  - Input layer: Features after preprocessing.
  - Hidden Layer 1: 64 neurons, ReLU activation.
  - Hidden Layer 2: 32 neurons, ReLU activation.
  - Output Layer: 1 neuron, Sigmoid activation.

- Compilation:

  - Optimizer: Adam with a learning rate of 0.01.
  - Loss function: Binary Crossentropy for Classification, Mean Average Score for Regression.

- Training:

  - Early stopping to prevent overfitting.
  - TensorBoard for logging and visualization.

## Results

- The model achieves competitive accuracy on the test set, with performance logged using TensorBoard.
- Users can interact with the Streamlit app to make predictions and view probabilities.

## Additional Comments
I wrote this code while learning how to implement ANNs, if you require the dataset used, open an issue on this repository. 
