import requests
import pandas as pd
import time
import random
import json
import os
from Utils.utils import get_content_from_yaml, load_config

def get_air_quality_df(station_id, specie):
    """
    Fetches air quality data for a given station and pollutant (specie).
    
    Args:
        station_id (str): The ID of the station to query data for.
        specie (str): The pollutant type ('pm10' or 'pm25').

    Returns:
        pd.DataFrame: A dataframe containing datetime and pollutant data.
    """
    # API endpoint for fetching historic daily data
    api_url = f'https://airnet.waqi.info/airnet/sse/historic/daily/{station_id}?specie={specie}'
    response = requests.get(api_url)
    daily_data = []
    if response.status_code == 200:
        lines = response.text.splitlines()
        for line in lines:
            if line.startswith("data: "):
                json_data = line[6:]
                try:
                    data = eval(json_data)
                    filtred_data = {'datetime': data['day'], specie: data['median']}
                    daily_data.append(filtred_data)
                except:
                    pass
    return pd.DataFrame(daily_data)

def get_api_data(stations):
    """
    Fetches and processes air quality data for multiple stations.

    Args:
        stations (dict): A dictionary mapping station IDs to station names.

    Returns:
        pd.DataFrame: A combined dataframe with data from all stations.
    """
    air_quality_data = {}

    for station_id, station_name in stations.items():
        delay = random.uniform(50, 100)
        time.sleep(delay)  # Pause to avoid API limits

        df_pm10 = get_air_quality_df(station_id, "pm10")
        df_pm25 = get_air_quality_df(station_id, "pm25")

        df_station = pd.merge(df_pm10, df_pm25, on='datetime', how='inner')
        df_station.set_index("datetime", inplace=True)

        df_station.columns = pd.MultiIndex.from_product([[station_name], df_station.columns])

        air_quality_data[station_name] = df_station

    final_df = pd.concat(air_quality_data.values(), axis=1)
    final_df.sort_index(inplace=True)

    return final_df

def main():
    """
    Main function to coordinate data fetching, processing, and saving.
    """
    print("Process start")
    config = load_config('process/config.yaml')
    
    communes = config['General']['communes']

    stations = get_content_from_yaml('communes.yaml')

    with open('data/General/air_quality.json', 'r', encoding='utf-8') as file:
        air_quality = json.load(file)

    for commune in communes:
        print(f"Obtaining data for the commune: {commune}")
        df = get_api_data(stations[commune])
        os.makedirs(f'data/{commune}', exist_ok=True)
        df.to_json(f'data/{commune}/pm10_pm25.json', orient='split', date_format='iso')
        last_row = df.iloc[-1]
        last_row_cleaned = last_row.dropna()
        for i, col in enumerate(last_row_cleaned.index):
            if i % 2 == 0:
                last_row_cleaned[col] = last_row_cleaned[col] / 2
        air_quality[commune] = last_row_cleaned.max()
    with open('data/General/air_quality.json', 'w', encoding='utf-8') as file:
        json.dump(air_quality, file, ensure_ascii=False, indent=4)
    print("Process complete")

if __name__ == "__main__":
    """process for obtaining pm10 and pm2.5 values for each station"""
    main()
