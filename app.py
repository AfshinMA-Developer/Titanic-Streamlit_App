import joblib
import numpy as np
import pandas as pd

import streamlit as st
from utils import columns, PreProcessor

# Load ML model
try:
    lr_model = joblib.load('./models/lrpipe.joblib')
except Exception as e:
    st.error('⚠️ The model file does not exist. Run titanic_model file') 

# Set Title
st.title('Would you make it out alive if you were on board the Titanic 🚢?')

# Get variables
passengerid = st.text_input(label="Input Passenger ID", value='8585') 
name  = st.text_input(label="Input Passenger Name", value='Afshin MA')
pclass = st.selectbox(label="Choose class", options=[1,2,3])
sex = st.radio(label='Choose gender', options=['male','female'])
age = st.slider(label="Choose age", max_value=100, min_value=0)
sibsp = st.slider(label="Choose siblings", max_value=10, min_value=0)
parch = st.slider(label="Choose parch", max_value=10, min_value=0)
ticket = st.text_input(label="Input Ticket Number", value="8585") 
fare = st.number_input(label="Input Fare Price", max_value=1000, min_value=0)
cabin = st.text_input(label="Input Cabin", value="C52") 
embarked = st.select_slider(label="Did they Embark?", options=['S','C','Q'])

# Define Predict function
def predict(): 
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)
    prediction = lr_model.predict(X)
    if prediction[0] == 1: 
        st.success('Passenger Survived 🙂')
    else: 
        st.error('Passenger did not Survive 😢') 

# Predict 
predict_btn = st.button('Predict', on_click=predict)