import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

st.title('Fuel Price Insights: Analyzing Historical Trends And Predicting Future Price Movements')

# Define product IDs and corresponding names
product_ids = [1, 2, 6]
id_to_name = {1: 'Euro-Super 95', 2: 'Automotive Gas Oil', 6: 'Residual Fuel Oil'}

# Load data and preprocess
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['SURVEY_DATE'] = pd.to_datetime(data['SURVEY_DATE'], format='%d-%m-%Y')
    data.set_index('SURVEY_DATE', inplace=True)
    return data

# Sidebar
st.sidebar.title("Fuel Product Selection")
selected_product_id = st.sidebar.selectbox("Select Fuel Product:", product_ids, format_func=lambda x: id_to_name[x])

# Dataset snapshot
dataset_visible = st.sidebar.checkbox("Show Dataset Snapshot", value=True)

# Load data file
file_path = './weekly_fuel_prices_all_data_from_2005_to_20221102-1.csv'  # Update this with your data path
data = load_data(file_path)

# Filter and clean data based on selected product
filtered_data = data[data['PRODUCT_ID'] == selected_product_id].copy().interpolate(method='linear')

# Display dataset snapshot if visible
if dataset_visible:
    st.header('Dataset Snapshot')
    st.write(filtered_data.head(10))

# Generate fuel price plot button
generate_plot = st.sidebar.button("Generate Fuel Price Plot")

# Fuel price trends plot
if generate_plot:
    st.header('Fuel Price Trends')
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['PRICE'], mode='lines', name='Price'))
    fig_price.update_layout(xaxis_title='Date', yaxis_title='Price', title='Fuel Price Trends')
    st.plotly_chart(fig_price)
    # Hide dataset snapshot after generating plot
    dataset_visible = False  # Update visibility status

# Forecasting controls in sidebar
st.sidebar.header('Forecasting')
n_weeks = st.sidebar.number_input('Enter number of weeks to forecast:', min_value=1, value=52)
generate_sarima = st.sidebar.button('Generate SARIMA Forecast')
generate_arima = st.sidebar.button('Generate ARIMA Forecast')

# Helper functions for forecasting
def perform_sarima_forecasting(product_data, n_weeks, use_cached_model=True):
    model_path = 'sarima_model.pkl'
    if use_cached_model and os.path.exists(model_path):
        # Load the model from the disk
        with open(model_path, 'rb') as file:
            sarima_model = joblib.load(file)
    else:
        # Fit a new model
        sarima_model = SARIMAX(product_data, order=(1, 1, 0), seasonal_order=(1, 1, 0, 52))
        sarima_results = sarima_model.fit()
        # Save the model to disk
        with open(model_path, 'wb') as file:
            joblib.dump(sarima_results, file)
    
    # Load the model and forecast
    forecast = sarima_model.get_forecast(steps=n_weeks)
    forecast_index = pd.date_range(product_data.index[-1] + pd.DateOffset(weeks=1), periods=n_weeks, freq='W-MON')
    return forecast_index, forecast.predicted_mean

def perform_arima_forecasting(product_data, n_weeks):
    arima_model = ARIMA(product_data, order=(1, 1, 0))
    arima_results = arima_model.fit()
    forecast = arima_results.forecast(steps=n_weeks)
    forecast_index = pd.date_range(product_data.index[-1] + pd.DateOffset(weeks=1), periods=n_weeks, freq='W-MON')
    return forecast_index, forecast

# SARIMA Forecast Plot
if generate_sarima:
    with st.spinner('Calculating SARIMA Forecast...'):
        forecast_index, forecast_values = perform_sarima_forecasting(filtered_data['PRICE'], n_weeks)
        fig_sarima = go.Figure()
        fig_sarima.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['PRICE'], mode='lines', name='Historical'))
        fig_sarima.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='SARIMA Forecast'))
        fig_sarima.update_layout(title='SARIMA Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_sarima)

# ARIMA Forecast Plot
if generate_arima:
    with st.spinner('Calculating ARIMA Forecast...'):
        forecast_index, forecast_values = perform_arima_forecasting(filtered_data['PRICE'], n_weeks)
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['PRICE'], mode='lines', name='Historical'))
        fig_arima.add_trace(go.Scatter(x=forecast_index, y=forecast_values, mode='lines', name='ARIMA Forecast'))
        fig_arima.update_layout(title='ARIMA Forecast', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_arima)