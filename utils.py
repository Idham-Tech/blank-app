import streamlit as st
from plotly import graph_objs as go 
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from tensorflow.keras.models import load_model

def plot_actual_data(dates, data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=data, name='NonMigas'))
    fig.layout.update(title_text='Non-Oil and Gas', xaxis_rangeslider_visible=True, hovermode = 'x')
    st.plotly_chart(fig)

def plot_train_gru(dates_train, train_act, train_pred):
    figgrutrain = go.Figure()
    figgrutrain.layout.update(title_text=('Actual and Predicted With GRU MODEL (Train)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figgrutrain.add_trace(go.Scatter(x=dates_train, y=train_act, name='Actual Value'))
    figgrutrain.add_trace(go.Scatter(x=dates_train, y=train_pred, name='Predicted Low Price'))
    st.plotly_chart(figgrutrain)

def plot_predict_gru( dates_test, test_act, test_pred):
    figgrutest = go.Figure()
    figgrutest.layout.update(title_text=('Actual and Predicted With GRU MODEL (Test)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figgrutest.add_trace(go.Scatter(x=dates_test, y=test_act, name='Actual Value'))
    figgrutest.add_trace(go.Scatter(x=dates_test, y=test_pred, name='Predicted Low Price'))
    st.plotly_chart(figgrutest)

def plot_train_xgboost(pre_shift_data, y_pre_shift, y_train_pred_pre_shift_full):
    y_pre_shift2 = y_pre_shift.flatten()
    y_train_pred_pre_shift_full2 = y_train_pred_pre_shift_full.flatten()
    figxgbtrain = go.Figure()
    figxgbtrain.layout.update(title_text=('Actual and Predicted With XGBoost MODEL (Train)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figxgbtrain.add_trace(go.Scatter(x=pre_shift_data.index, y=y_pre_shift2, name='Actual Value'))
    figxgbtrain.add_trace(go.Scatter(x=pre_shift_data.index, y=y_train_pred_pre_shift_full2, name='Predicted Low Price'))
    st.plotly_chart(figxgbtrain)

def plot_predict_xgboost( post_shift_data, y_post_shift, y_pred_post_shift):
    y_post_shift2 = y_post_shift.flatten()
    y_pred_post_shift2 = y_pred_post_shift.flatten()
    figxgbtest = go.Figure()
    figxgbtest.layout.update(title_text=('Actual and Predicted With XGBoost MODEL (Test)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    figxgbtest.add_trace(go.Scatter(x=post_shift_data.index, y=y_post_shift2, name='Actual Value'))
    figxgbtest.add_trace(go.Scatter(x=post_shift_data.index, y=y_pred_post_shift2, name='Predicted Low Price'))
    st.plotly_chart(figxgbtest)
    
def plot_predict_hybrid(data, y_test_actual, final_preds):
    y_test_actual2 = y_test_actual.flatten()
    final_preds2 = final_preds.flatten()
    fighybtest = go.Figure()
    fighybtest.layout.update(title_text=('Actual and Predicted With GRU-XGBoost MODEL (Test)'), xaxis_rangeslider_visible=True, hovermode = 'x')
    fighybtest.add_trace(go.Scatter(x=data.index[-len(y_test_actual2):], y=y_test_actual2, name='Actual Value'))
    fighybtest.add_trace(go.Scatter(x=data.index[-len(y_test_actual2):], y=final_preds2, name='Predicted Low Price'))
    st.plotly_chart(fighybtest)

def forcast_gru(model, data, seq_length, scaler):
    last_sequence = data['NonMigas'][-seq_length:].values.reshape((1, seq_length, 1))
    next_month_prediction = model.predict(last_sequence)
    next_month_prediction = scaler.inverse_transform(next_month_prediction)
    st.write('🙌🏻Next month export price forecast:', next_month_prediction)

def forcast_xgboost(post_shift_data, data, model_post_shift, scaler):
    # Forecasting the next month's value
    last_date = post_shift_data.index[-1]
    next_month = last_date + pd.DateOffset(months=1)

    # Prepare features for the next month prediction
    next_month_features = {
        'month': [next_month.month],
        'year': [next_month.year]
    }

    for lag in range(1, 13):
        next_month_features[f'lag_{lag}'] = [data['NonMigas'].iloc[-lag]]

    next_month_df = pd.DataFrame(next_month_features)

    # Predict the next month's NonMigas value
    next_month_scaled_prediction = model_post_shift.predict(next_month_df)
    next_month_prediction = scaler.inverse_transform(next_month_scaled_prediction.reshape(-1, 1))

    st.write('🙌🏻Next month export price forecast:', next_month_prediction)

def forcast_hybrid(stacked_test, meta_learner, scaler):
    latest_stacked_test = stacked_test[-1].reshape(1, -1)  # Reshape to a 2D array

    # Predict the next data point
    next_month_pred = meta_learner.predict(latest_stacked_test)

    # Inverse transform the next month's prediction
    next_month_pred = scaler.inverse_transform(next_month_pred.reshape(-1, 1))

    # Output the next month's prediction
    st.write('🙌🏻Next month export price forecast:', next_month_pred)

def evaluation(y_test_inv, test_predictions):
    test_mae = mean_absolute_error(y_test_inv, test_predictions)
    test_mape = mean_absolute_percentage_error(y_test_inv, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predictions))
    st.write('MAE:', test_mae, '  \nRMSE:', test_rmse, '  \nMAPE:', test_mape)

def gru_models(model):
    # Preprocess the data
    dataset = pd.read_csv('/workspaces/blank-app/Keseluruhan (Coba-coba) NonMigas.csv')
    dataset['date'] = pd.to_datetime(dataset['date'], format='%Y %B')
    dataset = dataset.sort_values('date')
    scaler = MinMaxScaler()
    dataset['NonMigas'] = scaler.fit_transform(dataset[['NonMigas']])
    normalizedData = dataset

    # Create sequences for training the GRU model
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 12  # Using 12 months (1 year) for each sequence
    X, y = create_sequences(dataset['NonMigas'], seq_length)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]  
    
    # Reshape the data to fit the GRU input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Make predictions on the training set
    train_predictions = model.predict(X_train)

    # Inverse transform the training predictions and actual values
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    
    # Create DataFrame for training actual and predicted values
    df_train = pd.DataFrame({   
        'Date': dataset['date'].iloc[seq_length:split_index + seq_length].reset_index(drop=True),
        'Actual': y_train_inv.flatten(),
        'Predicted': train_predictions.flatten()
    })  
    # Make predictions on the test set
    test_predictions = model.predict(X_test)

    # Inverse transform the predictions and actual values
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Create DataFrame for testing actual and predicted values
    df_test = pd.DataFrame({
        'Date': dataset['date'].iloc[split_index + seq_length:].reset_index(drop=True),
        'Actual': y_test_inv.flatten(),
        'Predicted': test_predictions.flatten()
    })

    return df_train, df_test, y_test_inv, test_predictions, normalizedData, seq_length, scaler

def model_xgboost():
    # Load the dataset
    file_path = '/workspaces/blank-app/Keseluruhan (Coba-coba) NonMigas.csv'
    data = pd.read_csv(file_path)

    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y %B')

    # Set the date column as the index
    data.set_index('date', inplace=True)

    # Normalize the NonMigas component
    scaler = MinMaxScaler()
    data['NonMigas'] = scaler.fit_transform(data[['NonMigas']])

    # Feature Engineering: Adding lag features, rolling mean, and standard deviation
    data['month'] = data.index.month
    data['year'] = data.index.year

    # Adding lag features
    for lag in range(1, 13):  # Adding lag features for the past 12 months
        data[f'lag_{lag}'] = data['NonMigas'].shift(lag)

    # Dropping rows with NaN values (due to lag and rolling calculations)
    data.dropna(inplace=True)

    # Splitting the data into pre-shift and post-shift sets
    pre_shift_data = data[:2017]
    post_shift_data = data['2018':]

    # Features and target variable for pre-shift data
    X_pre_shift = pre_shift_data.drop(columns=['NonMigas'])
    y_pre_shift = pre_shift_data['NonMigas']

    # Split pre-shift data into training and validation sets
    X_train_pre_shift, X_val_pre_shift, y_train_pre_shift, y_val_pre_shift = train_test_split(
        X_pre_shift, y_pre_shift, test_size=0.2, random_state=42)

    # Features and target variable for post-shift data
    X_post_shift = post_shift_data.drop(columns=['NonMigas'])
    y_post_shift = post_shift_data['NonMigas']

    # Split post-shift data into training and validation sets
    X_train_post_shift, X_val_post_shift, y_train_post_shift, y_val_post_shift = train_test_split(
        X_post_shift, y_post_shift, test_size=0.2, random_state=42)

    # Training the XGBoost model on pre-shift data
    model_pre_shift = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, alpha=0.3, lambda_=0.3)
    model_pre_shift.fit(X_train_pre_shift, y_train_pre_shift, eval_set=[(X_train_pre_shift, y_train_pre_shift), (X_val_pre_shift, y_val_pre_shift)], verbose=True)
  
    # Making predictions on pre-shift train data
    y_train_pred_pre_shift_full = model_pre_shift.predict(X_pre_shift)
    y_train_pred_pre_shift_full = scaler.inverse_transform(y_train_pred_pre_shift_full.reshape(-1, 1))
      
    # Training the XGBoost model on post-shift data with evaluation set
    model_post_shift = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, alpha=0.3, lambda_=0.3)
    model_post_shift.fit(X_train_post_shift, y_train_post_shift, eval_set=[(X_train_post_shift, y_train_post_shift), (X_val_post_shift, y_val_post_shift)], verbose=True)

    # Making predictions on post-shift test data
    y_pred_post_shift = model_post_shift.predict(X_post_shift)

    # Inverse transform the test predictions and actual values
    y_pred_post_shift = scaler.inverse_transform(y_pred_post_shift.reshape(-1, 1))
    y_post_shift = scaler.inverse_transform(y_post_shift.values.reshape(-1, 1))
    y_pre_shift = scaler.inverse_transform(y_pre_shift.values.reshape(-1, 1))

    y_train_pred_pre_shift_full = model_pre_shift.predict(X_pre_shift)
    y_train_pred_pre_shift_full = scaler.inverse_transform(y_train_pred_pre_shift_full.reshape(-1, 1))

    return data, model_post_shift, post_shift_data, y_post_shift, y_pred_post_shift, pre_shift_data, y_pre_shift, y_train_pred_pre_shift_full, scaler

def model_hybrid(model):
    # Load the dataset
    file_path = '/workspaces/blank-app/Keseluruhan (Coba-coba) NonMigas.csv'
    data = pd.read_csv(file_path)

    # Convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format='%Y %B')

    # Set the date column as the index
    data.set_index('date', inplace=True)

    # Feature Scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Preparing the data for GRU model
    def create_dataset(dataset, look_back=12):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 12

    # Creating GRU datasets
    X_gru, y_gru = create_dataset(data_scaled, look_back)
    X_gru = np.reshape(X_gru, (X_gru.shape[0], X_gru.shape[1], 1))

    # Splitting the data into training and testing sets for GRU
    train_size = int(len(X_gru) * 0.8)
    X_train_gru, X_test_gru = X_gru[:train_size], X_gru[train_size:]
    y_train_gru, y_test_gru = y_gru[:train_size], y_gru[train_size:]

    # Making predictions
    train_predict_gru = model.predict(X_train_gru)
    test_predict_gru = model.predict(X_test_gru)

    # Feature Engineering for XGBoost
    data['NonMigas'] = data_scaled[:, 0]  # Replace the NonMigas column with scaled values
    data['month'] = data.index.month
    data['year'] = data.index.year

    # Adding lag features
    for lag in range(1, 13):  # Adding lag features for the past 12 months
        data[f'lag_{lag}'] = data['NonMigas'].shift(lag)

    # Dropping rows with NaN values (due to lag and rolling calculations)
    data.dropna(inplace=True)

    # Splitting the data into pre-shift and post-shift sets
    pre_shift_data = data[:2017]
    post_shift_data = data['2018':]

    # Features and target variable for pre-shift data
    X_pre_shift = pre_shift_data.drop(columns=['NonMigas'])
    y_pre_shift = pre_shift_data['NonMigas']

    # Features and target variable for post-shift data
    X_post_shift = post_shift_data.drop(columns=['NonMigas'])
    y_post_shift = post_shift_data['NonMigas']

    # Split post-shift data into training and validation sets
    X_train_post_shift, X_val_post_shift, y_train_post_shift, y_val_post_shift = train_test_split(
        X_post_shift, y_post_shift, test_size=0.2, random_state=42)

    # Training the XGBoost model on pre-shift data
    model_pre_shift = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, alpha=0.3, lambda_=0.3)
    model_pre_shift.fit(X_pre_shift, y_pre_shift, verbose=True)

    # Making predictions on post-shift test data
    y_train_pred_pre_shift = model_pre_shift.predict(X_pre_shift)
    
    # Training the XGBoost model on post-shift data with evaluation set
    model_post_shift = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=3, alpha=0.3, lambda_=0.3)
    model_post_shift.fit(X_train_post_shift, y_train_post_shift, eval_set=[(X_train_post_shift, y_train_post_shift), (X_val_post_shift, y_val_post_shift)], verbose=True)

    # Making predictions on post-shift test data
    y_pred_post_shift = model_post_shift.predict(X_post_shift)

    # Ensure the lengths match by truncating the longer set
    min_len_train = min(len(train_predict_gru), len(y_train_pred_pre_shift))
    min_len_test = min(len(test_predict_gru), len(y_pred_post_shift))

    train_predict_gru = train_predict_gru[:min_len_train]
    y_train_pred_pre_shift = y_train_pred_pre_shift[:min_len_train]
    y_train_gru = y_train_gru[:min_len_train]

    test_predict_gru = test_predict_gru[:min_len_test]
    y_pred_post_shift = y_pred_post_shift[:min_len_test]
    y_test_gru = y_test_gru[:min_len_test]

    # Stacking the predictions
    stacked_train = np.vstack((train_predict_gru.flatten(), y_train_pred_pre_shift)).T
    stacked_test = np.vstack((test_predict_gru.flatten(), y_pred_post_shift)).T

    # Try different meta-learners
    meta_learner = LinearRegression()

    meta_learner.fit(stacked_train, y_train_gru)
    final_preds = meta_learner.predict(stacked_test)

    # Inverse transform the final predictions and actual values
    final_preds = scaler.inverse_transform(final_preds.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test_gru.reshape(-1, 1))
    return data, y_test_actual, final_preds, meta_learner, stacked_test

def write_gru():
    st.write('Evaluation of GRU Model')
def write_xgboost():
    st.write('Evaluation of XGBoost Model')
def write_hybrid():
    st.write('Evaluation of hybrid GRU-XGBoost Model')
def data_act():
    st.subheader('Data Actual')
def visual_data():
    st.subheader('Visualizations Data')
def visual_trainval():
    st.subheader('Visualizations Train and Validation Loss')
def visual_actpred_datatrain():
    st.subheader('Visualizations Actual and Prediction Data Train')
def visual_actpred_data():
    st.subheader('Visualizations Actual and Prediction Data')
def write_evaluation():
    st.subheader('Evaluation')
def write_forecast():
    st.subheader('Forecasting')

def proccess(option):
    # load model
    if option == 'GRU':
        model = tf.keras.models.load_model('modelgru.h5')
        df_train, df_test, y_test_inv, test_predictions, normalizedData, seq_length, scaler= gru_models(model)
            
        visual_actpred_data()
        plot_train_gru(df_train['Date'], df_train['Actual'], df_train['Predicted'])
        plot_predict_gru(df_test['Date'], df_test['Actual'], df_test['Predicted'])
            
    
        #evaluation
        write_evaluation()
        evaluation(y_test_inv, test_predictions)
        
        #forcasting
        forcast_gru(model, normalizedData, seq_length, scaler)

    elif option == 'XGBoost': 
        data, model_post_shift, post_shift_data, y_post_shift, y_pred_post_shift, pre_shift_data, y_pre_shift, y_train_pred_pre_shift_full, scaler = model_xgboost()
        
        visual_actpred_data()
        plot_train_xgboost(pre_shift_data, y_pre_shift, y_train_pred_pre_shift_full)
        plot_predict_xgboost(post_shift_data, y_post_shift, y_pred_post_shift)

        #evaluation
        write_evaluation()
        evaluation(y_post_shift,  y_pred_post_shift)

        #forcasting
        forcast_xgboost(post_shift_data, data, model_post_shift, scaler)        

    else:
        model = tf.keras.models.load_model('modelgru.h5')
        df_train, df_test, y_test_inv, test_predictions, normalizedData, seq_length, scaler= gru_models(model)
        data, model_post_shift, post_shift_data, y_post_shift, y_pred_post_shift, pre_shift_data, y_pre_shift, y_train_pred_pre_shift_full, scaler = model_xgboost()
        data2, y_test_actual, final_preds, meta_learner, stacked_test = model_hybrid(model)

        visual_actpred_data()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('GRU')
            plot_predict_gru(df_test['Date'], df_test['Actual'], df_test['Predicted'])
        with col2:
            st.write('XGBoost')
            plot_predict_xgboost(post_shift_data, y_post_shift, y_pred_post_shift)
        with col3:
            st.write('GRU-XGBoost')
            plot_predict_hybrid(data2, y_test_actual, final_preds)

        #evaluation
        write_evaluation()
        col4, col5, col6 = st.columns(3)
        with col4:
            st.write('GRU')
            evaluation(y_test_inv, test_predictions)
        with col5:
            st.write('XGBoost')
            evaluation(y_post_shift,  y_pred_post_shift)
        with col6:
            st.write('GRU-XGBoost')
            evaluation(y_test_actual,  final_preds)

        #forcasting
        write_forecast()
        col7, col8, col9 = st.columns(3)
        with col7:
            st.write('GRU')
            forcast_gru(model, normalizedData, seq_length, scaler)
        with col8:
            st.write('XGBoost')
            forcast_xgboost(post_shift_data, data, model_post_shift, scaler)
        with col9:
            st.write('GRU-XGBoost')
            forcast_hybrid(stacked_test, meta_learner, scaler)  