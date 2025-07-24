import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

# Configure Streamlit page
st.set_page_config(page_title="Stock Price Prediction", layout="centered")

# Title and subtitle
st.title("📈 AI Stock Price Prediction")
st.write("Forecast stock prices using Prophet with interactive visualization.")

# Constants
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Ticker selection
stocks = ("AAPL", "GOOG", "NFLX", "TSLA", "MSFT", "META", "AMZN")
selected_stock = st.selectbox("Select a stock for prediction", stocks)

n_years = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Function to load data safely
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    if data.empty:
        st.error(f"No data found for {ticker}. Please try another stock.")
        st.stop()
    data.reset_index(inplace=True)
    return data

# Load and display data
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

# Debug: Show available columns
st.write("Columns in loaded data:", data.columns.tolist())

# Safe preparation for Prophet
try:
    df_train = data[['Date', 'Close']].copy()
    df_train.columns = ['ds', 'y']
    df_train = df_train.dropna(subset=['ds', 'y'])
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
    df_train = df_train.dropna(subset=['y'])
except Exception as e:
    st.error(f"❌ Error preparing data for Prophet: {e}")
    st.stop()

if df_train.empty:
    st.error("❌ No valid data available for training Prophet. Please try another stock or adjust date range.")
    st.stop()

st.write(f"✅ Using {len(df_train)} rows of clean data for Prophet.")

# Train Prophet model with caching
@st.cache_resource
def train_prophet(df):
    m = Prophet()
    m.fit(df)
    return m

m = train_prophet(df_train)

# Forecast
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

st.success("✅ App is running cleanly without errors and is ready for sharing and deployment.")
