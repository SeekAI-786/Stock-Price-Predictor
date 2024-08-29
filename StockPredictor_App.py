import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

current_date = datetime.now().strftime('%Y-%m-%d')

st.title('Stock Price Predictor Using ML')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')


start = "2018-01-01"
end = current_date

df = yf.download(user_input, start= start, end = end)


st.subheader('Data From 2018 to Current Day')
st.write(df.describe())

st.subheader('Closing Price vs  Time Chart')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close, 'b')
st.pyplot(fig)


st.subheader('Closing Price vs  Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot( ma100, 'r',label = 'MA 100')
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs  Time Chart with 100 MA vs Time Chart with 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot( ma100, 'r',label = 'MA 100')
plt.plot( ma200, 'g', label = 'MA 200')
plt.plot(df.Close, 'b', label ='Closing Price')
plt.legend()
st.pyplot(fig)


#Training and Testing
data_tr = pd.DataFrame(df['Close'][0:int(len(df) * 0.70) ])
data_test = pd.DataFrame(df['Close'][int(len(df) * 0.70) : int(len(df))])


scaler = MinMaxScaler(feature_range = (0,1))
data_tr_arr = scaler.fit_transform(data_tr)


# Load Saved Model
model = load_model('keras_model.h5')

# Testing
past_100_days = data_tr.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])
    
    
x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)
scalor = scaler.scale_

scale_factor = 1 / scalor[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Orginal')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_predicted, 'r' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2)
