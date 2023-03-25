import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn import preprocessing

st.write("# Predictive Maintenance ")
scaler = preprocessing.MinMaxScaler()
#uploading train_df to complete the preprocessing
train_df = pd.read_csv('train_df.csv')
#making three columns for input 
col1, col2, col3 = st.columns(3)

#Taking input from the customers
var1 = col1.number_input("Number of Cycles",
    step=1e-6,
    format="%.5f")
var2 = col2.number_input("Operational Setting 1",
    step=1e-6,
    format="%.5f")
var3 = col3.number_input("Operational Setting 2",
    step=1e-6,
    format="%.5f")
var4 = col1.number_input("Sensor 2",
    step=1e-6,
    format="%.5f")
var5 = col2.number_input("Sensor 3",
    step=1e-6,
    format="%.5f")
var6 = col3.number_input("Sensor 4",
    step=1e-6,
    format="%.5f")
var7 = col1.number_input("Sensor 6",
    step=1e-6,
    format="%.5f")
var8 = col2.number_input("Sensor 7",
    step=1e-6,
    format="%.5f")
var9 = col3.number_input("Sensor 8",
    step=1e-6,
    format="%.5f")
var10 = col1.number_input("Sensor 9",
    step=1e-6,
    format="%.5f")
var11 = col2.number_input("Sensor 11",
    step=1e-6,
    format="%.5f")
var12 = col3.number_input("Sensor 12",
    step=1e-6,
    format="%.5f")
var13 = col1.number_input("Sensor 13",
    step=1e-6,
    format="%.5f")
var14 = col2.number_input("Sensor 15",
    step=1e-6,
    format="%.5f")
var15 = col3.number_input("Sensor 17",
    step=1e-6,
    format="%.5f")
var16 = col1.number_input("Sensor 20",
    step=1e-6,
    format="%.5f")
var17 = col2.number_input("Sensor 21",
    step=1e-6,
    format="%.5f")
var18 = col3.number_input("Max Cycle",
    step=1e-6,
    format="%.5f")

#making a data frame of the given input

test_df=pd.DataFrame([[var1,var2,var3,var4,var5,var6,var7,var8,var9,var10,var11,var12,var13,var14,var15,var16,var17,var18]],columns=["cycles","op_setting1","op_setting2","s2","s3","s4","s6","s7","s8","s9","s11","s12","s13","s15","s17","s20", "s21","maxcycles"])
#normalize the input
train_df = scaler.fit_transform(train_df)
test_df.loc[:,'op_setting1':'s21'] = scaler.transform(test_df.loc[:,'op_setting1':'s21'])


#loading the saved model
model = keras.models.load_model('ANN_model')
Test_X = test_df.values[:,0:17]
output = model.predict(Test_X)
test_df['output'] = output[0][0]
#doing some processing to convert fraction to cycles
def convertFractionToCycle(test_df):
  return(test_df['cycles'] / (1-test_df['output']))
test_df['predictedmaxcycles'] = convertFractionToCycle(test_df)
OUTPUT = test_df['predictedmaxcycles'] - test_df['maxcycles']
#displaying the output
if st.button('Predict'):
	st.write(f'<p style="font-size:26px;"> RUL: {int(OUTPUT[0])} Cycles</p>',unsafe_allow_html=True)
