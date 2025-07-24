import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# App configuration
st.set_page_config(page_title="Stock Price Prediction", layout="centered")
st.title("📈 AI Stock Price Prediction")
st.write("Use Prophet to forecast stock prices interactively.")

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Use default ticker list to avoid missing CSV issues
stocks = ("AAPL", "GOOG", "NFLX", "TSLA", "MSFT", "META", "AMZN")
selected_stock = st.selectbox("Select stock for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Use st.cache_data instead of st.cache
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load and display raw data
data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data... Done!")

st.subheader(f"Raw data for {selected_stock}")
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

plot_raw_data()

# Forecasting using Prophet
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Cache Prophet model
@st.cache_resource
def train_prophet(df):
    m = Prophet()
    m.fit(df)
    return m

m = train_prophet(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display forecast data
st.subheader("Forecast data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot forecast
st.subheader("Forecast plot")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

# Plot forecast components
st.subheader("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.info("✅ App ready for Streamlit Cloud deployment without CSV dependency or cache deprecation issues.")
