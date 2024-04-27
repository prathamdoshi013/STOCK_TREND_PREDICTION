# Stock Price Prediction using LSTM

This project aims to predict stock prices using Long Short-Term Memory (LSTM) networks. It utilizes historical stock price data downloaded from Yahoo Finance, preprocesses the data, trains an LSTM model, and then predicts future stock prices.

## Requirements

Make sure you have the following libraries installed:

- numpy
- pandas
- yfinance
- scikit-learn
- matplotlib
- tensorflow
- streamlit

You can install them using pip:

```
pip install numpy pandas yfinance scikit-learn matplotlib tensorflow streamlit
```

## Usage

1. Clone this repository:

```
git clone https://github.com/your-username/stock-price-prediction.git
```

2. Navigate to the project directory:

```
cd stock-price-prediction
```

3. Run the Streamlit app:

```
streamlit run app.py
```

4. Enter the stock ticker symbol and select the number of years of data you want to analyze using the slider.

5. The app will display the historical closing price chart and the predicted closing price chart based on the LSTM model.

## File Description

- `app.py`: Streamlit app for user interaction and displaying visualizations.
- `main.py`: Main file containing functions for downloading data, training the LSTM model, and predicting future prices.
- `keras_model.h5`: Pre-trained LSTM model saved in h5 format.
- `README.md`: This file providing information about the project.
- `requirement.txt` : This file will contain neccesary libraries to be installed
