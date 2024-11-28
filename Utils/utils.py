import json
from pandas import DataFrame
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from typing import Dict
import yaml

def load_config(yaml_file: str) -> Dict[str, any]:
    """
        Load and validate a YAML configuration file.

        Parameters:
            yaml_file (str): The path to the YAML configuration file.

        Returns:
            dict: The parsed configuration from the YAML file.
    """
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    if not config['Enable']:
        print("The process is disabled in the configuration.")
        exit()
    return config

def get_content_from_yaml(yaml_file: str) -> Dict[str, any]:
    """
        Reads and extracts the content from a YAML file.

        Parameters:
            yaml_file (str): The path to the YAML file to be read.

        Returns:
            dict or list: The content of the YAML file parsed into a Python dictionary or list,
                          depending on the structure of the YAML file.
    """
    with open(yaml_file, 'r') as file:
        stations = yaml.safe_load(file)
    return stations

def merge_pm_and_weather(commune: str, station: str) -> DataFrame:
    """
        Merges air pollution data (PM10 and PM2.5) and weather data for a given commune and station.

        Parameters:
            - commune (str): Name of the commune (subdirectory within the 'data' folder) containing the data files.
            - station (str): Name of the station used to extract specific PM10/PM2.5 data.

        Returns:
            - merged_df (pd.DataFrame): Merged DataFrame containing the combined weather and PM data.
    """
    with open(f'data/{commune}/pm10_pm25.json', 'r', encoding='utf-8') as pm_file:
        pm_json = json.load(pm_file)

    with open(f'data/{commune}/weather.json', 'r', encoding='utf-8') as weather_file:
        weather_json = json.load(weather_file)
    
    columns = pd.MultiIndex.from_tuples(pm_json['columns'])
    pm_data = pd.DataFrame(pm_json['data'], index=pm_json['index'], columns=columns)
    df1 = pm_data[station]
    df2 = pd.DataFrame(data=weather_json["data"], columns=weather_json["columns"])

    df1.index = pd.to_datetime(df1.index)
    df2['datetime'] = pd.to_datetime(df2['datetime'])

    common_dates = df1.index.intersection(df2['datetime'])

    df1_filtered = df1.loc[common_dates]
    df2_filtered = df2[df2['datetime'].isin(common_dates)]

    merged_df = pd.merge(df2_filtered, df1_filtered, left_on='datetime', right_index=True)
    return merged_df

def filter(df: DataFrame) -> DataFrame:
    """
        Cleans and processes a DataFrame by converting datetime columns, extracting
        temporal features, and removing unnecessary columns.

        Parameters:
            df (pandas.DataFrame): The input DataFrame

        Returns:
            pandas.DataFrame: A cleaned DataFrame
        """
    if 'datetime' in df:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.weekday
    # Drop unnecessary columns
    df = df.drop(columns=['datetime', 'index'], errors='ignore')
    return df.dropna()

def create_dense_model(input_shape: int) -> Sequential:
    """
        Creates a dense neural network model using TensorFlow's Keras API.

        Parameters:
            input_shape : int
                The number of features in the input data. This determines the shape of the input layer.

        Returns:
            model : tf.keras.Sequential
                A compiled Keras Sequential model consisting of:
                    - A dense layer with 64 units and ReLU activation.
                    - A dense layer with 32 units and ReLU activation.
                    - A dense output layer with 2 units (intended for predicting PM10 and PM2.5).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)  # Sortie pour pm10 et pm25
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_lstm_model(input_shape: int) -> Sequential:
    """
        Creates and compiles an LSTM-based neural network model for regression or binary classification tasks.

        Parameters:
            input_shape (int): The number of features in the input data.
                               This represents the dimensionality of each input vector.

        Returns:
            model : tf.keras.Sequential
                A compiled TensorFlow Keras Sequential model with the following layers:
                    - LSTM layer with 64 units and ReLU activation.
                    - Dense layer with 32 units and ReLU activation.
                    - Dense output layer with 2 units (e.g., for binary classification or 2-output regression).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(1, input_shape)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model