import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os
import datetime
import itertools
from scipy.stats import percentileofscore, norm

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
import keras.backend as K



def data_loading(filenameDM, filenameCP, P_ind, P_paths, start_year, time_resolution):
    # Function that loads data for a certain DMA, as this is defined by the indices of the timeseries recorded
    # at the entrance of the DMA (filenameDM) and at the critical point (filenameCP). Data are extracted from
    # a certain year (start_year) onwards and at a pre-defined time resolution (time_resolution), e.g., 15-min.
    
    # custom_date_parser = lambda x: datetime.datetime.strptime(x, "%d/%m/%Y %H:%M")
    custom_date_parser = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

    # Read time-series from sensor at DMA entry
    df = pd.read_csv(filenameDM, delimiter=',', index_col=0, parse_dates=[0], date_parser=custom_date_parser)
    # df.drop(df.columns[[2, 3, 4, 5, 6]], axis=1, inplace=True)  # keep only 1-min flow and pressure data
    df = df.iloc[:,[0,1]]  # keep only 1-min flow and pressure data
    df.columns = ['P', 'Q']  # rename columns with meaningful names

    # Read time-series from sensor at DMA critical point
    dfCP = pd.read_csv(filenameCP, delimiter=',', index_col=0, parse_dates=[0], date_parser=custom_date_parser)
    # dfCP.drop(dfCP.columns[[1, 2, 3, 4]], axis=1, inplace=True)  # keep only 1-min pressure data
    dfCP = dfCP.iloc[:,0]  # keep only 1-min pressure data
    dfCP.columns = ['P_CP']  # rename column with meaningful names

    # Merge time-series from two sensors
    df = df.assign(P_CP=dfCP)
    
    if len(P_ind) > 0:  # Multiple P signals
        for i in range(len(P_ind)):
            df_temp = pd.read_csv(P_paths[i], delimiter=',', index_col=0, parse_dates=[0], date_parser=custom_date_parser)
            df_temp = df_temp.iloc[:,0]  # keep only 1-min pressure data
            df_temp.columns = [P_ind[i]]  # rename column with meaningful name
            df[P_ind[i]] = df_temp
    
    # Keep time-series since start year
    if start_year == 2016:
        df = df[(df.index >= datetime.datetime.strptime("1/1/2016","%d/%m/%Y")) & (df.index <= datetime.datetime.strptime("25/3/2022","%d/%m/%Y"))]
    elif start_year == 2018:
        df = df[(df.index >= datetime.datetime.strptime("1/2/2018","%d/%m/%Y")) & (df.index <= datetime.datetime.strptime("25/3/2022","%d/%m/%Y"))]
    else:  # start_year == 2020
        df = df[(df.index >= datetime.datetime.strptime("12/3/2020","%d/%m/%Y")) & (df.index <= datetime.datetime.strptime("25/3/2022","%d/%m/%Y"))]
    
    if time_resolution == 15:
        df = df.resample('15Min', origin='start_day').mean()  # calculate 15-min mean values
    elif time_resolution == 30:
        df = df.resample('30Min', origin='start_day').mean()  # calculate 30-min mean values
    else: # time_resolution == 60
        df = df.resample('1H', origin='start_day').mean()  # calculate 1-hour mean values
    
    # Drop rows with missing data
    df = df.dropna()
    df = df.loc[~(df<=0).any(axis=1)]
    
    print('Data loading completed')

    return df

### ____________________________________________________________________________________________________________

def data_features(df, time_resolution, verbose=False):
    # Function that creates additional features in the dataframe, besides those recorded (Q, P, P @ C.P.). These
    # features include: Month, Weekday (with special care for bank holidays and possibly new year's), the
    # consecutive Deltas (x_i-1 - x_i) for Discharge and the Pressure signals, a boolean Burst index and finally 
    # the Minute of the day.
    
    df['Month'] = df.index.month  # extracting month from timestamp
    df['Weekday'] = df.index.weekday
    
    # Apply function 0.2 / (1 - 0.8*cos(x)) where x is the weekday index (0=Sunday, 6=Saturday)
    # Assign 1 to Weekend, and smaller values (close to zero) to all the working days
    df['TimeFeature1'] = 0.2 / (1 - 0.8 * np.cos(np.pi * ((df.index.weekday + 1) % 7) / 3))

    bank_holidays = np.array([
        '1/1/2016', '25/3/2016', '28/3/2016', '2/5/2016', '30/5/2016', '29/8/2016', '25/12/2016', '26/12/2016', #2016
        '1/1/2017', '14/4/2017', '17/4/2017', '1/5/2017', '29/5/2017', '28/8/2017', '25/12/2017', '26/12/2017', #2017
        '1/1/2018', '30/3/2018', '2/4/2018', '7/5/2018', '28/5/2018', '27/8/2018', '25/12/2018', '26/12/2018', #2018
        '1/1/2019', '19/4/2019', '22/4/2019', '6/5/2019', '27/5/2019', '26/8/2019', '25/12/2019', '26/12/2019', #2019
        '1/1/2020', '10/4/2020', '13/4/2020', '8/5/2020', '25/5/2020', '31/8/2020', '25/12/2020', '26/12/2020', '28/12/2020', #2020
        '1/1/2021', '2/4/2021', '5/4/2021', '3/5/2021', '31/5/2021', '30/8/2021', '25/12/2021', '26/12/2021', '27/12/2021', '28/12/2021' #2021
        ])
    
    for j in range(len(bank_holidays)):
        rows_boolean = (df.index >= datetime.datetime.strptime(bank_holidays[j],"%d/%m/%Y")) & (df.index < datetime.datetime.strptime(bank_holidays[j],"%d/%m/%Y") + datetime.timedelta(days=1))
        rows = [i for i, val in enumerate(rows_boolean) if val]
        df.iloc[rows, -1] = 1  # assigning 1 to bank holidays
        df['Weekday'].iloc[rows] = 7  # assigning 7 to bank holidays    

    df['Burst'] = np.zeros(len(df))
    df['Minute'] = (df.index.hour * 60 + df.index.minute)  # calculate minute of day
    
    if verbose==True:
        display(df)
    
    return df

### ____________________________________________________________________________________________________________

def initial_visualization(df):
    # Function that visualizes the intitially loaded data Q, P and P @ C.P.

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(13, 5))
    f.tight_layout(h_pad=3)
    
    ax1.plot(df['Q'], label='Discharge');
    ax1.set_title('Discharge');
    ax1.set_xlim([df.index[0], df.index[-1]])
    ax1.set_ylabel('Q [m3/s]')
    
    ax2.plot(df['P'], label='Pressure');
    ax2.set_title('Pressure');
    ax2.set_xlim([df.index[0], df.index[-1]])
    ax2.set_ylabel('Dh [m]')
    
    ax3.plot(df['P_CP'], label='Pressure @ CP');
    ax3.set_title('Pressure @ CP');
    ax3.set_xlim([df.index[0], df.index[-1]])
    ax3.set_ylabel('Dh [m]');
    
    plt.suptitle(f'Time-series visualization', y=1.05);
    
    return

### ____________________________________________________________________________________________________________

def load_leak_job_records(directory_path, DMA_name, start_year, first_index, verbose=False):
    # Function that aggregates labeled leak job information from multiple years and records and returns a
    # dataframe with the indexed leak jobs, their detection and repair time.
    
    # __________ Leak jobs from 2021 to 2022 __________
    filename_2021_2022 = directory_path + "\Leak job records" + '/' + 'Leak Job Data 2021-2022.xls'
    job_record_2021_2022 = pd.read_excel(filename_2021_2022, index_col=0)  # Download leak job record
    bursts_2021_2022 = job_record_2021_2022.loc[job_record_2021_2022['Region Description'] == DMA_name]  # Isolate jobs in one DMA
    bursts_2021_2022 = bursts_2021_2022.drop(bursts_2021_2022.index[np.where(bursts_2021_2022 == '\xa0')[0]])  # drop non-repaired leak jobs
    for record in [bursts_2021_2022]:
        record.drop(record.columns[[0, 2]], axis=1, inplace=True)  # Drop unecessary information
        record.columns = ['Detection Date', 'Job End Date', 'Description']  # rename columns with meaningful names

        for i in record.columns[0:2]:
            record[i] = pd.to_datetime(record[i], format="%d/%m/%Y %H:%M").dt.strftime("%Y/%m/%d %H:%M")
    
    # __________ Leak jobs from 2020 __________
    filename_2020 = directory_path + "\Leak job records" + '/' + 'Leak Job Data 2020.xls'
    job_record_2020 = pd.read_excel(filename_2020, index_col=0)  # Download leak job record
    bursts_2020 = job_record_2020.loc[job_record_2020['Region Description'] == DMA_name]  # Isolate jobs in one DMA
    bursts_2020 = bursts_2020.drop(bursts_2020.index[np.where(bursts_2020 == '\xa0')[0]])  # drop non-repaired leak jobs
    for record in [bursts_2020]:
        record.drop(record.columns[[0, 2]], axis=1, inplace=True)  # Drop unecessary information
        record.columns = ['Detection Date', 'Job End Date', 'Description']  # rename columns with meaningful names

        for i in record.columns[0:2]:
            record[i] = pd.to_datetime(record[i], format="%d/%m/%Y %H:%M").dt.strftime("%Y/%m/%d %H:%M")
    
    # __________ Leak jobs from 2018 to 2020 __________
    filename_2018_2020 = directory_path + "\Leak job records" + '/' + 'Leak Job Data 2018-2020.xlsx'
    job_record_2018_2020 = pd.read_excel(filename_2018_2020, index_col=0)  # Download leak job record
    bursts_2018_2020 = job_record_2018_2020.loc[job_record_2018_2020['Region Description'] == DMA_name]  # Isolate jobs in one DMA
    bursts_2018_2020 = bursts_2018_2020.drop(bursts_2018_2020.index[np.where(bursts_2018_2020 == '\xa0')[0]])  # drop non-repaired leak jobs
    for record in [bursts_2018_2020]:
        record.drop(record.columns[0], axis=1, inplace=True)  # Drop unecessary information
        record.columns = ['Detection Date', 'Job End Date', 'Description']  # rename columns with meaningful names

        for i in record.columns[0:2]:
            record[i] = pd.to_datetime(record[i], format="%d/%m/%Y %H:%M").dt.strftime("%Y/%m/%d %H:%M")

        record["Job End Date"] = pd.to_datetime(record["Job End Date"], format="%Y/%m/%d 00:00").dt.strftime("%Y/%m/%d 23:59")
    
    # __________ Leak jobs up to 2018 __________
    filename_2016_2018 = directory_path + "\Leak job records" + '/' + 'Leak Job Data 2008-2018.xlsx'
    job_record_2016_2018 = pd.read_excel(filename_2016_2018, index_col=0)  # Download leak job record
    bursts_2016_2018 = job_record_2016_2018.loc[job_record_2016_2018['Region Description'] == DMA_name]  # Isolate jobs in one DMA
    bursts_2016_2018 = bursts_2016_2018[bursts_2016_2018['Job Created Date'] >= datetime.datetime.strptime("1/1/2016","%d/%m/%Y")]  # Filter out bursts prior to 2016
    bursts_2016_2018 = bursts_2016_2018.drop(bursts_2016_2018.index[np.where(bursts_2016_2018 == '\xa0')[0]])  # drop non-repaired leak jobs
    for record in [bursts_2016_2018]:
        record.drop(record.columns[3], axis=1, inplace=True)  # Drop unecessary information
        record.columns = ['Description', 'Detection Date', 'Job End Date']  # rename columns with meaningful names
        # record = record[['Detection Date', 'Job End Date', 'Description']]  # rearrange columns

        for i in record.columns[1:]:
            record[i] = pd.to_datetime(record[i], format="%Y-%m-%d").dt.strftime("%Y/%m/%d 00:00")

        record["Job End Date"] = pd.to_datetime(record["Job End Date"], format="%Y/%m/%d 00:00").dt.strftime("%Y/%m/%d 23:59")
    
    # __________ Aggregate leak jobs __________
    if start_year == 2016:
        bursts = bursts_2021_2022.append(bursts_2020).append(bursts_2018_2020).append(bursts_2016_2018)
    elif start_year == 2018:
        bursts = bursts_2021_2022.append(bursts_2020).append(bursts_2018_2020)
    else:  # start_year == 2020
        bursts = bursts_2021_2022.append(bursts_2020)
    
    # Drop nan values in burst record
    bursts = bursts.dropna()
    bursts = bursts.sort_values(by='Detection Date')
    
    # Drop bursts that took place prior to the start of the dataset
    boolean_drop = np.array(np.zeros(len(bursts)),dtype='bool')
    for j in range(len(bursts)):
        if (datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") >= first_index) != True:
            boolean_drop[j] = True
    bursts = bursts[boolean_drop == False]
    
    if verbose==True:
        display('Bursts:', bursts)
    
    return bursts

### ____________________________________________________________________________________________________________

def leak_flagging(df, bursts, day_num, verbose=False):
    # Function that sets the boolean "Burst" feature in the dataframe to 1, when a burst (leak or otherwise) has
    # taken place and it is within within "day_num" days (e.g., 1 day) from when the client became aware of it. 
    # Note that the client is becomes aware of the leak at the detection time.
    
    for j in range(len(bursts)):
        # Seperate non-leaky data from data that were recorded within one week (i.e., 7 days) of a known leak.
        # A known leak is defined by two times, detection and time and end job time.
        rows_boolean = (df.index >= datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") - datetime.timedelta(days=day_num)) & (df.index <= datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M"))
        rows = [i for i, val in enumerate(rows_boolean) if val]
        df['Burst'][rows] = 1
    
    if verbose==True:
        print(f'Shape of dataframe entries with burst: {df[df["Burst"] == 1].shape}')
        print(f'Shape of dataframe entries without burst: {df[df["Burst"] == 0].shape}')

    return df

### ____________________________________________________________________________________________________________

def extract_rolling_weeks(df, feature_list, P_ind, num_rolling_days, time_win_len, num_features, verbose=True):
    # Function that extracts 7-day rolling feature datasets of length time_win_len
    
    if len(P_ind) > 0:
        feature_list = feature_list[:-2] + P_ind + feature_list[-2:]
        
    bool_array = np.zeros(len(df)).astype(dtype=bool)
    for i in range(0, int(len(bool_array) - time_win_len)):
        if df.index[i + time_win_len - 1] == df.index[i] + datetime.timedelta(days=num_rolling_days):
            bool_array[i] = True

    data = np.zeros((sum(bool_array), time_win_len, num_features))
    for i, stamp0 in enumerate(df.index[bool_array]):
        stamp1 = stamp0 + datetime.timedelta(days=num_rolling_days)
        data[i] = df.loc[stamp0:stamp1][feature_list]
        
    if verbose==True:
        print('Data shape:', data.shape)
    
    return data, bool_array

### ____________________________________________________________________________________________________________

def scaling3D(x_train, x_val, x_test, num_features, verbose=False):
    # Function that scales 3D arrays based on a Min-Max Scaler fit on the training data only.
    
    x_train_scaled = []; x_val_scaled = []; x_test_scaled = []
    
    scaler_list = []
    for i in range(num_features):
        scaler_name = 'scaler' + str(i)
        locals()[scaler_name] = MinMaxScaler()
        x_train_scaled.append(np.array([locals()[scaler_name].fit_transform(x_train[:, :, i])])[0])
        x_val_scaled.append(np.array([locals()[scaler_name].transform(x_val[:, :, i])])[0])
        x_test_scaled.append(np.array([locals()[scaler_name].transform(x_test[:, :, i])])[0])
        scaler_list.append(locals()[scaler_name])
    
    x_train = np.stack(x_train_scaled, axis=2)
    x_val = np.stack(x_val_scaled, axis=2)
    x_test = np.stack(x_test_scaled, axis=2)
        
    print('Data scaling completed')
    
    if verbose==True:
        print('x_train:', x_train.shape, ' ', 'x_val:', x_val.shape, ' ', 'x_test:', x_test.shape)

    return x_train, x_val, x_test, scaler_list

### ____________________________________________________________________________________________________________

def rescaling3D(x_train, x_val, x_test_partial, num_features, verbose=False):
    # Function that scales 3D arrays based on a Min-Max Scaler fit on the training data only.
    
    x_train_val_test_partial = np.append(np.append(x_train, x_val, axis=0), x_test_partial, axis=0)
    
    x_train_val_test_scaled = []
    
    scaler_list = []
    for i in range(num_features):
        scaler_name = 'scaler' + str(i)
        locals()[scaler_name] = MinMaxScaler()
        x_train_val_test_scaled.append(np.array([locals()[scaler_name].fit_transform(x_train_val_test_partial[:, :, i])])[0])
        scaler_list.append(locals()[scaler_name])
    
    x_train_val_test = np.stack(x_train_val_test_scaled, axis=2)
        
    print('Data rescaling completed')
    
    if verbose==True:
        print('x_train_val_test:', x_train_val_test.shape)

    return x_train_val_test

### ____________________________________________________________________________________________________________

def reshape3D_to_2D(x_train, x_val, x_test):
    # Function that reshapes 3D to 2D arrays, so that these can be fed to an neural network.
    
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1] * x_val.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    
    print('3D to 2D reshaping completed')
        
    return x_train, x_val, x_test

### ____________________________________________________________________________________________________________

def one_resampling(x, time_win_len, weekday_time_resolution, time_resolution):
    # Function that resamples Weekday and SHP features of one subset to a custom (weekday_time_resolution) resolution
    
    x_TimeFeature1 = x[:,:time_win_len]
    x_TimeFeature2 = x[:,time_win_len:int(2*time_win_len)]
    x_weekday3 = x[:,int(2*time_win_len):]

    for index, y in enumerate([x_TimeFeature1, x_TimeFeature2, x_weekday3]):
        i = 0
        alpha = np.array([y[:,i*int(weekday_time_resolution / time_resolution):(i+1)*int(weekday_time_resolution / time_resolution)].mean(axis=1)]).T
        for i in range(1, int(np.ceil(time_win_len / (weekday_time_resolution / time_resolution)))):
            alpha = np.append(alpha, np.array([y[:,i*int(weekday_time_resolution / time_resolution):(i+1)*int(weekday_time_resolution / time_resolution)].mean(axis=1)]).T, axis=1)
        if index == 0:
            x_okay = alpha
        else:
            x_okay = np.append(x_okay, alpha, axis=1)
    return x_okay

### ____________________________________________________________________________________________________________

def feature_resampling(x_train, x_val, x_test, time_win_len, weekday_time_resolution, time_resolution, verbose=True):
    # Function that resamples Weekday and SHP features of all subsets to a custom (weekday_time_resolution) resolution
    
    x_train = one_resampling(x_train, time_win_len, weekday_time_resolution, time_resolution)
    x_val = one_resampling(x_val, time_win_len, weekday_time_resolution, time_resolution)
    x_test = one_resampling(x_test, time_win_len, weekday_time_resolution, time_resolution)
    
    if verbose==True:
        print('Feature resampling completed')
    
    return x_train, x_val, x_test

### ____________________________________________________________________________________________________________

def create_AE(x_train, verbose=False):
    # Function that creates the Autoencoder
    
    # Input
    i = layers.Input(shape=(x_train.shape[1]), name='Input')
    # Encoding
    x = layers.Dense(64, activation="relu", name="Encoder1")(i)
    x = layers.Dense(32, activation="relu", name="Encoder2")(x)
    # Code
    x = layers.Dense(16, activation="relu", name="bottleneck")(x)
    # Decoding
    x = layers.Dense(32, activation="relu", name="Decoder1")(x)
    x = layers.Dense(64, activation="relu", name="Decoder2")(x)
    # Output
    o = layers.Dense(x_train.shape[1], activation="relu", name="Output")(x)
    
    autoencoder = Model(inputs=[i], outputs=[o])
    autoencoder.compile(optimizer='adam', loss='mse')
        
    if verbose==True:
        autoencoder.summary()
    
    return autoencoder

### ____________________________________________________________________________________________________________

def create_ED(x1_train, x2_train, verbose=False):
    # Function that creates the Encoder-Decoder
    
    # Input
    i1 = layers.Input(shape=(x1_train.shape[1]), name='Input1')
    # Encoding
    x1 = layers.Dense(64, activation="relu", name="Encoder1")(i1)
    x1 = layers.Dense(32, activation="relu", name="Encoder2")(x1)
    
    # Input of Weekday features
    i2 = layers.Input(shape=(x2_train.shape[1]), name='Input2')
    x2 = layers.Dense(32, activation="relu", name="Branch_Encoder1")(i2)
    x2 = layers.Dense(32, activation="relu", name="Branch_Encoder2")(x2)
    
    x1_2 = layers.add([x1, x2])
    
    # Code
    x = layers.Dense(16, activation="relu", name="bottleneck")(x1_2)
    # Decoding
    x = layers.Dense(32, activation="relu", name="Decoder1")(x)
    x = layers.Dense(64, activation="relu", name="Decoder2")(x)
    # Output
    o = layers.Dense(x1_train.shape[1], activation="relu", name="Output")(x)
    
    autoencoder = Model(inputs=[i1, i2], outputs=[o])
    autoencoder.compile(optimizer='adam', loss='mse')
        
    if verbose==True:
        autoencoder.summary()
    
    return autoencoder

### ____________________________________________________________________________________________________________

def create_LSTM_AE(x_train, time_win_len, verbose=False):
    # Function that creates the LSTM Autoencoder
    
    model = Sequential()
    model.add(layers.LSTM(units=32, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.RepeatVector(n=x_train.shape[1]))
    model.add(layers.LSTM(units=32, return_sequences=True))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.TimeDistributed(layers.Dense(units=x_train.shape[2])))
    model.compile(optimizer='adam', loss='mse')
    autoencoder = model
        
    if verbose==True:
        autoencoder.summary()
    
    return autoencoder

### ____________________________________________________________________________________________________________

# def my_loss(y_true, y_pred, n_fitted_features):
#     # Function that calculates the error between observed and predicted values taking into account that
#     # flow (Q) measurements have a weight of 50%, and pressure (P) measurements share the remaining 50% weight
    
#     # n_fitted_features = 3
#     weights = np.ones(n_fitted_features) * (0.5 / (n_fitted_features - 1))
#     weights[0] = 0.5
    
#     squared_weighted_difference  = tf.square(weights * (y_true - y_pred))
#     # squared_weighted_difference  = tf.square(y_true - y_pred)
#     return tf.reduce_mean(squared_weighted_difference, axis=-1)

# def loss(n_fitted_features):
#     def customLoss(y_true, y_pred):
#         return my_loss(y_true, y_pred, n_fitted_features)
#     return customLoss

### ____________________________________________________________________________________________________________

def create_LSTM(x_train, y_train, num_features, dropout_rate, verbose=False):
    # Function that creates the LSTM NN
    
    i1 = layers.Input(shape=(x_train[:,:-1,:-2].shape[1], x_train[:,:-1,:-2].shape[2]), name='Input1')  # n_samples, length, features
    x1 = layers.LSTM(units=16, dropout=dropout_rate, return_sequences=False, name='LSTM')(i1)

    i2 = layers.Input(shape=(2), name='Input2')  # n_samples, features
    x2 = layers.Dense(2, activation="relu")(i2)

    x1_x2 = layers.Concatenate()([x1, x2])  
    o = layers.Dense(int(num_features - 2), activation="relu")(x1_x2)

    autoencoder = Model(inputs=[i1, i2], outputs=[o])
    
    # n_fitted_features = x_train[:,:-1,:-2].shape[2]
    # model_loss = loss(n_fitted_features)
    # autoencoder.compile(optimizer='adam', loss=model_loss)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    if verbose==True:
        autoencoder.summary()
    
    return autoencoder

### ____________________________________________________________________________________________________________

def create_expanded_LSTM(x_train, y_train, num_features, dropout_rate, verbose=False):
    # Function that creates the LSTM NN
    
    i1 = layers.Input(shape=(x_train[:,:-1,:-2].shape[1], x_train[:,:-1,:-2].shape[2]), name='Input1')  # n_samples, length, features
    x1 = layers.LSTM(units=16, dropout=dropout_rate, return_sequences=False, name='LSTM')(i1)

    i2 = layers.Input(shape=(2), name='Input2')  # n_samples, features
    x2 = layers.Dense(2, activation="relu")(i2)

    x1_x2 = layers.Concatenate()([x1, x2])  
    o = layers.Dense(int(num_features - 2), activation="relu")(x1_x2)

    autoencoder = Model(inputs=[i1, i2], outputs=[o])
    
    # n_fitted_features = x_train[:,:-1,:-2].shape[2]
    # model_loss = loss(n_fitted_features)
    # autoencoder.compile(optimizer='adam', loss=model_loss)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    if verbose==True:
        autoencoder.summary()
    
    return autoencoder

### ____________________________________________________________________________________________________________

def transfer_learning(pre_trained_model, num_features, autoencoder):
    # Get weights of LSTM layer from pre-trained model
    weights_LSTM = pre_trained_model.layers[2].get_weights()
    inputs, hidden_units, bias = weights_LSTM

    # Expand weights matrices by duplicating pressure signals
    extended_inputs = inputs
    for i in range(num_features - 5):
        extended_inputs = np.append(extended_inputs, inputs[1].reshape((1, len(inputs[1]))), axis=0)

    # Set new model weights equal to expanded weights matrices
    autoencoder.layers[2].set_weights([extended_inputs, hidden_units, bias])

    # Get weights of output layer from pre-trained model
    weights_output = pre_trained_model.layers[5].get_weights()
    outputs, bias = weights_output

    # Expand weights matrices by duplicating pressure signals
    extended_outputs = outputs
    extended_bias = bias
    for i in range(num_features - 5):
        extended_outputs = np.append(extended_outputs, outputs[:, 1].reshape((len(outputs[:, 1]), 1)), axis=1)
        extended_bias = np.append(extended_bias, bias[1])

    # Set new model weights equal to expanded weights matrices
    autoencoder.layers[5].set_weights([extended_outputs, extended_bias])
    
    return autoencoder

### ____________________________________________________________________________________________________________

def train_AE(autoencoder, x_train, x_val, reduce_lr, er_stopping, verbose=0, batch_size=1024, epochs=100):
    # Function that fits AE to training data and checks convergence with validation data
    
    history = autoencoder.fit(x_train, x_train, 
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=[reduce_lr, er_stopping],
                              validation_data=(x_val, x_val),
                              shuffle=True)
    
    print('AE training completed')
    
    return autoencoder, history

### ____________________________________________________________________________________________________________

def train_ED(autoencoder, x1_train, x2_train, x1_val, x2_val, reduce_lr, er_stopping, verbose=0, batch_size=1024, epochs=100):
    # Function that fits AE to training data and checks convergence with validation data
    
    history = autoencoder.fit([x1_train, x2_train], x1_train, 
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=[reduce_lr, er_stopping],
                              validation_data=([x1_val, x2_val], x1_val),
                              shuffle=True)
    
    print('ED training completed')
    
    return autoencoder, history

### ____________________________________________________________________________________________________________

def train_LSTM(autoencoder, x_train, x_val, y_train, y_val, reduce_lr, er_stopping, verbose=0, verbose_print=False, batch_size=1024, epochs=100):
    # Function that fits LSTM NN to training data and checks convergence with validation data
    
    history = autoencoder.fit([x_train[:, :-1, :-2], x_train[:, -1, -2:]], y_train, 
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=[reduce_lr, er_stopping],
                              validation_data=([x_val[:, :-1, :-2], x_val[:,-1, -2:]], y_val),
                              shuffle=False)
    if verbose_print==True:
        print('LSTM training completed')
    
    return autoencoder, history

### ____________________________________________________________________________________________________________

def predict_AE_one(autoencoder, x):
    # Function that reconstructs subsets
    x_pred = autoencoder.predict(x)    
    return x_pred

def predict_AE(autoencoder, x_train, x_val, x_test):
    # Function that reconstructs training, validation and testing subsets
    
    x_train_pred = predict_AE_one(autoencoder, x_train)
    x_val_pred = predict_AE_one(autoencoder, x_val)
    x_test_pred = predict_AE_one(autoencoder, x_test)
    
    print('AE predicting completed')
    
    return x_train_pred, x_val_pred, x_test_pred

### ____________________________________________________________________________________________________________

def predict_LSTM_one(autoencoder, i1, i2):
    # Function that reconstructs subsets
    x_pred = autoencoder.predict([i1, i2])    
    return x_pred

def predict_LSTM(autoencoder, x_train, x_val, x_test):
    # Function that reconstructs training, validation and testing subsets
    
    x_train_pred = predict_LSTM_one(autoencoder, x_train[:, :-1, :-2], x_train[:, -1, -2:])
    x_val_pred = predict_LSTM_one(autoencoder, x_val[:, :-1, :-2], x_val[:, -1, -2:])
    x_test_pred = predict_LSTM_one(autoencoder, x_test[:, :-1, :-2], x_test[:, -1, -2:])
    
    print('AE predicting completed')
    
    return x_train_pred, x_val_pred, x_test_pred

### ____________________________________________________________________________________________________________

def predict_ED_one(autoencoder, x1, x2):
    # Function that reconstructs subsets
    x_pred = autoencoder.predict([x1, x2])    
    return x_pred

def predict_ED(autoencoder, x1_train, x1_val, x1_test, x2_train, x2_val, x2_test):
    # Function that reconstructs training, validation and testing subsets
    
    x_train_pred = predict_ED_one(autoencoder, x1_train, x2_train)
    x_val_pred = predict_ED_one(autoencoder, x1_val, x2_val)
    x_test_pred = predict_ED_one(autoencoder, x1_test, x2_test)
    
    print('AE predicting completed')
    
    return x_train_pred, x_val_pred, x_test_pred

### ____________________________________________________________________________________________________________

def MSE_calc_one(x, x_pred):
    # Function that calculates MSE between an original and a reconstructed dataset
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    MSE_x = mse(x, x_pred).numpy()
    
    return MSE_x

def MSE_calc(x_train, x_val, x_test, x_train_pred, x_val_pred, x_test_pred, verbose=False):
    # Function that calculates MSE between the original and the reconstructed datasets
    
    # mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    
    MSE_train = MSE_calc_one(x_train, x_train_pred)
    MSE_val = MSE_calc_one(x_val, x_val_pred)
    MSE_test = MSE_calc_one(x_test, x_test_pred)
    
    if verbose == True:
        print('------- MSE -------')
        print(f'Training:   {np.mean(MSE_train):.4f}')
        print(f'Validation: {np.mean(MSE_val):.4f}')
        print(f'Testing:    {np.mean(MSE_test):.4f}')
    
    return MSE_train, MSE_val, MSE_test

### ____________________________________________________________________________________________________________

def MSE_hist(MSE_train, MSE_val, MSE_test, percentile_value):
    # Function that plots MSE histograms of training, validation and testing subsets 
    
    mse_perc = np.percentile(MSE_val, percentile_value)
    
    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 5))
    f.tight_layout(h_pad=3)

    ax1.hist(MSE_train, color='royalblue', bins=100, alpha=0.65, label='training');
    ax1.axvline(mse_perc, color='r', label=f'{percentileofscore(MSE_train, mse_perc):.1f}-th percentile');
    ax1.set_title('Training');
    ax1.legend()
    ax1.set_ylabel('Instances');
    ax1.set_xlabel('MSE');

    ax2.hist(MSE_val, color='limegreen', bins=100, alpha=0.65, label='validation');
    ax2.axvline(mse_perc, color='r', label=f'{percentileofscore(MSE_val, mse_perc):.1f}-th percentile');
    ax2.set_title('Validation');
    ax2.legend()
    ax2.set_ylabel('Instances');
    ax2.set_xlabel('MSE');
    
    ax3.hist(MSE_test, color='coral', bins=100, alpha=0.65, label='test');
    ax3.axvline(mse_perc, color='r', label=f'{percentileofscore(MSE_test, mse_perc):.1f}-th percentile');
    ax3.set_title('Testing');
    ax3.legend()
    ax3.set_ylabel('Instances');
    ax3.set_xlabel('MSE');
    
    plt.suptitle(f'Reconstruction error', y=1.03);
    
    return mse_perc

### ____________________________________________________________________________________________________________

def Variable_MSE(df, percentile_value, train_val_indices, bool_array_train_val, time_win_len, x_val, hour_intervals, verbose=True):
    # Function that calculates the MSE threshold corresponding to the percentile_value for each of the 24 hours of the day, based
    # on the MSE distribution of the same hour in the validation subset
    
    # Isolate validation entries within the dataframe
    val_entries = (train_val_indices[np.where(bool_array_train_val)[0]] + time_win_len)[-x_val.shape[0]:]  # df indices of validation subset
    working_days = np.where(df['Weekday']<5)[0]
    weekends_holidays = np.where(df['Weekday']>=5)[0]

    MSE_perc = np.zeros((2, int(24 / hour_intervals)))
    for i, group_indices in enumerate([working_days, weekends_holidays]):
        for j in np.arange(0, 24, hour_intervals):
            val_entries_of_hour = np.intersect1d(val_entries, np.where((df.index.hour >= j) & (df.index.hour <= (j + int(hour_intervals - 1))))[0])
            val_entries_of_hour_and_day_group = np.intersect1d(val_entries_of_hour, group_indices)
            MSE_perc[i, int(j / hour_intervals)] = np.percentile(df.MSE.iloc[val_entries_of_hour_and_day_group], percentile_value)

    if verbose==True:
        plt.figure(figsize=(9, 4))
        plt.plot(np.arange(0, 25), np.hstack((np.repeat(MSE_perc[0, :], hour_intervals), MSE_perc[0, 0])), label='Working days')
        plt.plot(np.arange(0, 25), np.hstack((np.repeat(MSE_perc[1, :], hour_intervals), MSE_perc[1, 0])), label='Weekends & Holidays')
        plt.title(f'{percentile_value}-th percentile MSE threshold')
        plt.xlabel('Hour of day')
        plt.ylabel('MSE')
        plt.legend();
        plt.xticks(np.arange(0, 25, 2))
        plt.xlim([0, 24]);
    
    return MSE_perc, working_days, weekends_holidays, val_entries

### ____________________________________________________________________________________________________________

def MSE_plot(df, bursts, train_val_indices, test_indices, valid_array, time_win_len, MSE_perc, num_rolling_days):
    # Function that plots the Reconstruction Error for th entire recorded time-series, spanning all
    # (training, validation, testing 1 and testing 2) datasets
    
    plt.figure(figsize=(14, 4))
    plt.plot(df['MSE'].iloc[np.intersect1d(valid_array, train_val_indices)[0]:np.intersect1d(valid_array, train_val_indices)[-1]], color='royalblue')  # ) + time_win_len]
    plt.plot(df['MSE'].iloc[np.intersect1d(valid_array, test_indices)], color='coral')
    plt.plot(df['MSE_thres'], color='red', linestyle='-.', label='MSE threshold');
    # plt.axhline(mse_perc, color='red', linestyle='-.', label='MSE threshold');
    for j in range(len(bursts)):
        start = datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M")
        end = datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M")
        plt.axvspan(start, end, color='grey', alpha=0.2, lw=0)
    
    plt.xlim([df.index[0], df.index[-1]])
    # plt.ylim([0, 3 * mse_perc])
    plt.title('Reconstruction error')
    plt.xlabel(f'End of {num_rolling_days}-day rolling period');
    plt.legend(['training & validation', 'testing', 'MSE threshold', 'leak']);
    
    return

### ____________________________________________________________________________________________________________

def create_leak_free_record(df, bursts, time_win_len):
    # Function that creates a dataframe with all the leak-free periods of duration more than 2*time_win_len
    # 1 time_win_len for the delay in the recession of MSE
    # 1 time_win_len for our assumption that a leak job can be detected up to a week before the client
    
    non_bursts_start = np.hstack((np.array([0]), np.intersect1d(np.where(df['Burst'] == 1)[0] + 1, np.where(df['Burst'] == 0)[0])))
    non_bursts_end = np.intersect1d(np.where(df['Burst'] == 1)[0] - 1, np.where(df['Burst'] == 0)[0])
    if non_bursts_start[-1] > non_bursts_end[-1]:
        non_bursts_end = np.hstack((np.intersect1d(np.where(df['Burst'] == 1)[0] - 1, np.where(df['Burst'] == 0)[0]), np.array([len(df)])))

    if df.Burst.iloc[0] == 1:
        print('First entry is a labeled leak. Fix: non_bursts_start')

    # Save start and end indices of leak-free periods in dataframe
    non_bursts = pd.DataFrame(columns = bursts.columns[:2])
    non_bursts[non_bursts.columns[0]] = non_bursts_start
    non_bursts[non_bursts.columns[1]] = non_bursts_end

    # Drop leak-free periods with duration less than time_win_len (i.e., 7 days)
    non_bursts.drop(np.where((non_bursts.iloc[:,0] + 2 * time_win_len) > non_bursts.iloc[:,1])[0])
    
    # Shrink non bursts by time_win_len from both sides (beginning and end)
    non_bursts.iloc[:,0] = non_bursts.iloc[:,0] + time_win_len
    non_bursts.iloc[:,1] = non_bursts.iloc[:,1] - time_win_len
    
    return non_bursts

### ___________________________________________________________________________________________________________

def leak_job_count(A, B, df, bursts, BattLeDIM, horizon_time):
    # Function that counts recorded jobs between time points A and B
    
    count_num = 0
    detected_bursts = np.array([])
    detection_time = np.array([])
    for j in range(len(bursts)):
        # Condition 1: A before burst repair: leaks that have been repaired are ignored
        condition1 = (datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M") >= datetime.datetime.strptime(str(df.index[A]),"%Y-%m-%d %H:%M:%S"))
        if BattLeDIM == True:
            # Condition 2: A is after burst start
            condition2 = (datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") <= (datetime.datetime.strptime(str(df.index[A]),"%Y-%m-%d %H:%M:%S")))
        else:
            # Condition 2: A up to horizon_time before burst start: leaks that started a long time ago are ignored
            condition2 = (datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") <= (datetime.datetime.strptime(str(df.index[A]),"%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=horizon_time)))
        # Condition 3: A is after burst start
        condition3 = (datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") <= (datetime.datetime.strptime(str(df.index[A]),"%Y-%m-%d %H:%M:%S")))

        if condition1 & (condition2 | condition3):
            count_num = count_num + 1
            detected_bursts = np.append(detected_bursts, np.array(bursts.index[j]))
            detection_time = np.append(detection_time, convert_timedelta(datetime.datetime.strptime(str(df.index[A]),"%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M")))

    return count_num, detected_bursts, detection_time

### ____________________________________________________________________________________________________________

def burst_count(df_temp, bursts):
    bursts_boolean = np.zeros(len(bursts))
    for j in range(len(bursts)):
        cond1 = ((datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") > df_temp.index[0]) & (datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") < df_temp.index[-1]))
        cond2 = ((datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M") > df_temp.index[0]) & (datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M") < df_temp.index[-1]))
        if cond1 | cond2:
            bursts_boolean[j] = 1
        bursts_included = bursts.iloc[np.where(bursts_boolean==1)[0]]
    return bursts_included

### ____________________________________________________________________________________________________________

def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = seconds // 3600
    return (days * 24) + hours

### ____________________________________________________________________________________________________________

def performance_v3(df, bursts, train_indices, val_indices, test_indices, time_resolution, time_win_len, BattLeDIM, DMA_text, horizon_time=7, verbose=True):

    alarm_rows = np.where(df.MSE > df.MSE_thres)[0]
    df['Alarm'] = np.zeros(len(df))
    df['Alarm'][alarm_rows] = 1

    leak_detected_array = []
    leak_det_time_array = []

    for df_index, df_temp in enumerate([df.iloc[test_indices]]):
        bursts_included = burst_count(df_temp, bursts)

        ascend_indices = np.where((df_temp['Alarm'] == 0).ne((df_temp['Alarm'] == 0).shift(1)))[0][1:][::2]
        descend_indices = np.where((df_temp['Alarm'] == 0).ne((df_temp['Alarm'] == 0).shift(1)))[0][2:][::2]

        TP = 0; FP = 0; leak_detected_list = np.array([]); leak_det_time_list = np.array([])

        for i in range(len(descend_indices)):
            A = int(ascend_indices[i])
            B = int(descend_indices[i])
            count_leaks, det_ind, det_time = leak_job_count(A, B, df_temp, bursts_included, BattLeDIM, horizon_time)
            new_indices = np.nonzero(np.isin(det_ind, np.setdiff1d(det_ind, leak_detected_list)))[0]

            if count_leaks > 0:  # Detected leak
                if len(new_indices) > 0:  # New non-repeated detections
                    earliest_det_index = np.where(det_time == min(det_time[new_indices]))[0]
                    TP = TP + 1
                    leak_detected_list = np.append(leak_detected_list, det_ind[earliest_det_index])
                    leak_det_time_list = np.append(leak_det_time_list, det_time[earliest_det_index])
            else:
                FP = FP + 1

        leak_detected_array.append(leak_detected_list)
        leak_det_time_array.append(leak_det_time_list)
        
        burst_list = np.where(df_temp.Burst == 1)[0]
        no_burst_list = np.setdiff1d(np.where(df_temp.Burst == 0)[0], burst_list)
        alarm_list = np.where(df_temp.Alarm == 1)[0]
        no_alarm_list = np.where(df_temp.Alarm == 0)[0]
        
        # Value-based metrics
        TP_vb = len(np.intersect1d(burst_list, alarm_list))
        FP_vb = len(np.intersect1d(no_burst_list, alarm_list))
        TN_vb = len(np.intersect1d(no_burst_list, no_alarm_list))
        FN_vb = len(np.intersect1d(burst_list, no_alarm_list))
        Precision = TP_vb / (TP_vb + FP_vb)
        Recall = TP_vb / (TP_vb + FN_vb)
        Fallout = FP_vb / (FP_vb + TN_vb)

    if verbose==True:
        print(F'\n-------------- {DMA_text} ---------------')
        print(F'\n--------- event-based KPIs ---------')
        print(f'No. bursts = {len(bursts_included)} | Precision_e = {(100 * TP / (TP + FP)):.2f}% | Recall_e = {(100 * TP / len(bursts_included)):.2f}%')
        print(F'\n--------- value-based KPIs ---------')
        print(f'TPR = {(100 * Recall):.2f}% | FPR = {(100 * Fallout):.2f}% | Precision = {(100 * Precision):.2f}%')
    
    return leak_detected_array, leak_det_time_array

### ____________________________________________________________________________________________________________

def individual_leak_plot(j, df, bursts, MSE_perc, num_rolling_days, time_win_len, time_resolution):
    # Function that enables visualization of a period in time around a leak
    
    rows_burst_detection_boolean = (df.index >= datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M"))
    burst_detection_index = np.where(rows_burst_detection_boolean)[0][0]
    rows_burst_repair_boolean = (df.index >= datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M"))
    burst_repair_index = np.where(rows_burst_repair_boolean)[0][0]
    
    plot_start_index = max(burst_detection_index - time_win_len, 0)
    plot_end_index = min(burst_repair_index + time_win_len, len(df))
    
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=False, figsize=(12, 10))
    f.tight_layout(h_pad=3)

    ax1.plot(df['Q'].iloc[plot_start_index:plot_end_index], label='discharge');
    ax1.plot(df['Q'].iloc[max(plot_start_index - time_win_len, 0):plot_end_index].rolling(int(24 * (60 / time_resolution))).mean(), linestyle='-.', label='24-h rolling mean')
    ax1.set_title('Discharge');
    ax1.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    ax1.axvline(df.index[burst_detection_index], color='r', label='leak detection');
    ax1.axvline(df.index[burst_repair_index], color='g', label='leak repair');
    ax1.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
    ax1.legend(loc='upper right');

    ax2.plot(df['P'].iloc[plot_start_index:plot_end_index], label='Pressure');
    ax2.set_title('Pressure');
    ax2.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    ax2.axvline(df.index[burst_detection_index], color='r', label='leak detection');
    ax2.axvline(df.index[burst_repair_index], color='g', label='leak repair');
    ax2.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
    ax2.legend(loc='upper right');

    ax3.plot(df['P_CP'].iloc[plot_start_index:plot_end_index], label='Pressure @ CP');
    ax3.set_title('Pressure @ CP');
    ax3.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    ax3.axvline(df.index[burst_detection_index], color='r', label='leak detection');
    ax3.axvline(df.index[burst_repair_index], color='g', label='leak repair');
    ax3.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
    ax3.legend(loc='upper right');
    
    ax4.plot(df['MSE'].iloc[plot_start_index:plot_end_index], label='MSE');
    ax4.set_title('Reconstruction error');
    ax4.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    ax4.axvline(df.index[burst_detection_index], color='r', label='leak detection');
    ax4.axvline(df.index[burst_repair_index], color='g', label='leak repair');
    ax4.plot(df['MSE_thres'].iloc[plot_start_index:plot_end_index], color='orange', linestyle='-.', label='MSE threshold');
    # ax4.axhline(mse_perc, color='orange', linestyle='-.', label='MSE threshold');
    ax4.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
    ax4.legend(loc='upper right');
    
    ax5.plot(df['NFA'].iloc[max(plot_start_index - time_win_len, 0):plot_end_index].rolling(int(24 * (60 / time_resolution))).mean(), color='purple', label='% perc. of past 2 months');
    ax5.set_title('Night flow analysis');
    ax5.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    ax5.axvline(df.index[burst_detection_index], color='r', label='leak detection');
    ax5.axvline(df.index[burst_repair_index], color='g', label='leak repair');
    ax5.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
    ax5.legend(loc='upper right');
    ax5.set_xlabel('Time of observation');

    plt.suptitle(f'Leak job: {bursts.index[j]} registered at {bursts.iloc[j, 0]}\nDescription: {bursts.iloc[j, 2]}', y=1.07);
    
    return burst_detection_index, burst_repair_index, plot_start_index, plot_end_index

### ____________________________________________________________________________________________________________

def bursts_in_timewindow(bursts, start_time, end_time):
    # Function that returns indices of bursts within a defined time window, set by start_time & end_time
    
    bursts_included = np.zeros(len(bursts))
    
    for j in range(len(bursts)):
        if (datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M") 
            >= datetime.datetime.strptime(str(start_time),"%Y-%m-%d")) & (
            datetime.datetime.strptime(str(bursts.iloc[j, 1]),"%Y/%m/%d %H:%M") 
            <= datetime.datetime.strptime(str(end_time),"%Y-%m-%d")) | (
            (datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") 
            >= datetime.datetime.strptime(str(start_time),"%Y-%m-%d")) & (
            datetime.datetime.strptime(str(bursts.iloc[j, 0]),"%Y/%m/%d %H:%M") 
            <= datetime.datetime.strptime(str(end_time),"%Y-%m-%d"))):

            bursts_included[j] = 1
            
    return np.where(bursts_included == 1)[0]

### ____________________________________________________________________________________________________________

def data_explorer(df, bursts, MSE_perc, num_rolling_days, time_win_len, time_resolution, start_time, end_time, verbose=True):
    # Function that enables data visualization with custom date range and displays leaks in the timw window
    
    plot_start_index = max(np.where(df.index >= start_time)[0][0], 0)
    plot_end_index = min(np.where(df.index >= end_time)[0][0], len(df))
    
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=False, figsize=(12, 10))
    f.tight_layout(h_pad=3)

    ax1.plot(df['Q'].iloc[plot_start_index:plot_end_index], label='discharge');
    ax1.plot(df['Q'].iloc[max(plot_start_index - time_win_len, 0):plot_end_index].rolling(int(24 * (60 / time_resolution))).mean(), linestyle='-.', label='24-h rolling mean')
    ax1.set_title('Discharge');
    ax1.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])

    ax2.plot(df['P'].iloc[plot_start_index:plot_end_index], label='Pressure');
    ax2.set_title('Pressure');
    ax2.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])

    ax3.plot(df['P_CP'].iloc[plot_start_index:plot_end_index], label='Pressure @ CP');
    ax3.set_title('Pressure @ CP');
    ax3.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    
    ax4.plot(df['MSE'].iloc[plot_start_index:plot_end_index], label='MSE');
    ax4.set_title('Reconstruction error');
    ax4.plot(df['MSE_thres'].iloc[plot_start_index:plot_end_index], color='orange', linestyle='-.', label='MSE threshold');
    ax4.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    
    ax5.plot(df['NFA'].iloc[max(plot_start_index - time_win_len, 0):plot_end_index].rolling(int(24 * (60 / time_resolution))).mean(), color='purple', label='% perc. of past 2 months');
    ax5.set_title('Night flow analysis');
    ax5.set_xlim([df.index[plot_start_index], df.index[plot_end_index]])
    ax5.set_xlabel('Time of observation');
    
    burst_indices = bursts_in_timewindow(bursts, start_time, end_time)
    
    for j in range(len(burst_indices)):
                
        rows_burst_detection_boolean = (df.index >= datetime.datetime.strptime(str(bursts.iloc[burst_indices[j], 0]),"%Y/%m/%d %H:%M"))
        burst_detection_index = np.where(rows_burst_detection_boolean)[0][0]
        rows_burst_repair_boolean = (df.index >= datetime.datetime.strptime(str(bursts.iloc[burst_indices[j], 1]),"%Y/%m/%d %H:%M"))
        burst_repair_index = np.where(rows_burst_repair_boolean)[0][0]
        
        ax1.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
        ax2.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
        ax3.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
        ax4.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)
        ax5.axvspan(df.index[burst_detection_index], df.index[burst_repair_index], color='red', alpha=0.06, lw=0)

        ax1.axvline(df.index[burst_detection_index], color='r', label='leak detection');
        ax2.axvline(df.index[burst_detection_index], color='r', label='leak detection');
        ax3.axvline(df.index[burst_detection_index], color='r', label='leak detection');
        ax4.axvline(df.index[burst_detection_index], color='r', label='leak detection');
        ax5.axvline(df.index[burst_detection_index], color='r', label='leak detection');
        
        ax1.axvline(df.index[burst_repair_index], color='g', label='leak repair');
        ax2.axvline(df.index[burst_repair_index], color='g', label='leak repair');
        ax3.axvline(df.index[burst_repair_index], color='g', label='leak repair');
        ax4.axvline(df.index[burst_repair_index], color='g', label='leak repair');
        ax5.axvline(df.index[burst_repair_index], color='g', label='leak repair');
    
    ax1.legend(['discharge', '24-h rolling mean', 'leak detection', 'leak repair'], loc='upper right');
    ax2.legend(['D_Pressure', 'leak detection', 'leak repair'], loc='upper right');
    ax3.legend(['D_Pressure @ CP', 'leak detection', 'leak repair'], loc='upper right');
    ax4.legend(['MSE', 'leak detection', 'leak repair', 'MSE threshold'], loc='upper right');
    ax5.legend(['% perc. of past 2 months', 'leak detection', 'leak repair'], loc='upper right');
    
    plt.show()
    
    if verbose==True:
        if len(burst_indices > 0):
            print('\033[1m' + '\nLeak jobs visualized:\n' + '\033[0m')
        
        for j in range(len(burst_indices)):
            print(f'* Leak job no.{burst_indices[j]} with index: {bursts.index[burst_indices[j]]} registered at {bursts.iloc[burst_indices[j], 0]}\n  Description: {bursts.iloc[burst_indices[j], 2]}\n')

### ____________________________________________________________________________________________________________