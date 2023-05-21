import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 
import yfinance as yf
from keras.models import load_model



st.title('Stock Trend predictor')

user_input = st.text_input('Enter Stock Ticker','AAPL' )
script = yf.download(tickers=user_input,period='15y',interval='1d') 

#describing data
st.subheader('Data of last 15 years')
st.write(script.describe())

#visualization
st.subheader('closing price vs time chart')
fig = plt.figure(figsize=(17,7))
plt.plot(script.Close, 'b')
st.pyplot(fig)

st.subheader('Closing price vs time chart with 100MA')#100 ma
ma100 = script.Close.rolling(100).mean()
fig = plt.figure(figsize=(17,7))
plt.plot(ma100, 'g')
plt.plot(script.Close, 'b')
st.pyplot(fig) 

st.subheader('Closing price vs time chart with 100MA and 200MA')#200&100 ma
ma200 = script.Close.rolling(200).mean()
#ma100 = script.Close.rolling(100).mean()
fig = plt.figure(figsize=(17,7))
plt.plot(ma200, 'r')
plt.plot(ma100, 'g')
plt.plot(script.Close, 'b')
st.pyplot(fig) 

# splitting data into training and testing 

data_training = pd.DataFrame(script['Close'][0:int(len(script)*0.70)])
data_testing = pd.DataFrame(script['Close'][int(len(script)*0.70): int(len(script))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))  

data_training_array = scaler.fit_transform(data_training)

#load my model 
model = load_model('stock_model.h5')



#testing part
past_100_days = data_training.tail(100)
final_script = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_script)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# making prediction

y_predicted = model.predict(x_test) 
scaler.scale_

scale_factor = 1/scaler.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor
#final graph 

st.subheader('Prediction vs original')
fig2 = plt.figure(figsize=(17,7))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
