import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
import streamlit as st
from datetime import datetime, timedelta
import threading

# Function to get start date based on user input for number of years
def get_start_date(years):
    current_date = datetime.now()
    start_date = current_date - timedelta(days=years*365)
    return start_date.strftime("%Y-%m-%d")

# Function to download historical data from Yahoo Finance
@st.cache_data
def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Function to predict future prices
def predict_future_prices(scaler, model, X_test, days):
    future_prices = []
    current_data = X_test[-1]  # Get the most recent data
    for _ in range(days):
        predicted_price = model.predict(np.array([current_data]))  # Predict the next price
        future_prices.append(predicted_price[0, 0])  # Append the predicted price
        current_data = np.roll(current_data, -1, axis=0)  # Shift the data by one day
        current_data[-1] = predicted_price[0]  # Update the last element with the predicted price
    # Scale the predicted prices back to original scale
    future_prices = np.array(future_prices).reshape(-1, 1)
    future_prices = future_prices * scaler.scale_ + scaler.min_
    return future_prices

# Function to plot original vs predicted prices
def plot_prices(df, y_test, y_predicted):
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.index[-len(y_test):], y_test,'b', label = 'Original price')
    plt.plot(df.index[-len(y_predicted):], y_predicted, 'r',label = 'Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    return fig

# Streamlit app with Session State
def main():
    # Initialize session state
    if 'user_input' not in st.session_state:
        st.session_state.user_input = {
            'ticker': None,
            'years': None
        }
        
    # Initialize mutex lock
    if 'lock' not in st.session_state:
        st.session_state.lock = threading.Lock()

    # Display app title
    st.title('Stock Trend Prediction')

    # Display input fields only if ticker is not selected yet
    if st.session_state.user_input['ticker'] is None:
        st.subheader('EXAMPLE: SBIN.NS, RELIANCE.NS, TCS.NS, APPL, MSFT')
        st.session_state.user_input['ticker'] = st.text_input('Enter Stock Ticker')

    # Process user input
    if st.session_state.user_input['ticker']:
        st.session_state.user_input['years'] = st.slider('Select number of years:', 1, 10, 1)

        # Perform prediction if all inputs are provided
        if st.session_state.user_input['years']:
            with st.session_state.lock:
                start_date = get_start_date(st.session_state.user_input['years'])
                end_date = datetime.now().strftime("%Y-%m-%d")
                df = download_data(st.session_state.user_input['ticker'], start_date, end_date)

                st.subheader(f'Data from {start_date} to {end_date}')
                st.write(df.describe())

                st.subheader('Closing price vs Time chart')
                fig1 = plt.figure(figsize=(12,6))
                plt.plot(df.index, df.Close,'b')
                st.pyplot(fig1)

                # Use the entire data for training
                data_training = pd.DataFrame(df['Close'])

                # Scale the data
                scaler = MinMaxScaler(feature_range=(0,1))
                data_training_array = scaler.fit_transform(data_training)

                # Prepare the data for the LSTM model
                X = []
                y = []
                for i in range(100, data_training_array.shape[0]):
                    X.append(data_training_array[i-100: i])
                    y.append(data_training_array[i,0])
                X , y = np.array(X), np.array(y)

                # Load the model and predict the prices
                model = load_model('keras_model.h5')
                y_predicted = model.predict(X)

                # Reverse the scaling
                y_predicted = scaler.inverse_transform(y_predicted)

                # Plot the original vs predicted prices
                st.subheader('Original vs predicted price')
                fig2 = plot_prices(df, df['Close'].values, y_predicted.flatten())
                st.pyplot(fig2)

if __name__ == '__main__':
    main()
