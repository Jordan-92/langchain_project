import json
import requests
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from Utils.utils import load_config
from dotenv import load_dotenv

load_dotenv()

def resolve_dates(first_day, last_day):
    """Resolves 'Yesterday' or 'Max' dates into real dates"""
    actual_date = datetime.today()
    resolved_first_day = (actual_date - timedelta(days=1)).strftime('%Y-%m-%d') if first_day == "Yesterday" else first_day
    resolved_last_day = (actual_date + timedelta(days=15)).strftime('%Y-%m-%d') if last_day == "Max" else last_day
    return resolved_first_day, resolved_last_day

def get_weather_df(city, first_day, last_day):
    """Retrieves weather data for a given city"""
    api_key = os.getenv('WEATHER_API_KEY')
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{first_day}/{last_day}?unitGroup=us&include=days&key={api_key}&contentType=json"
    response = requests.get(url)
    print(response)
    if response.status_code == 200:
        weather_data = response.json()
        records = []
        days = weather_data['days']
        for day in days:
            record = {
                "datetime": day["datetime"],
                "temp": day.get("temp", None),
                "humidity": day.get("humidity", None),
                "precip": day.get("precip", None),
                "snow": day.get("snow", None),
                "windspeed": day.get("windspeed", None),
                "pressure": day.get("pressure", None),
                "uvindex": day.get("uvindex", None),
                "moonphase": day.get("moonphase", None)
            }
            records.append(record)
    else:
        print(f"Data recovery error : {response.status_code} - {response.text}")
        sys.exit(1)
    return pd.DataFrame(records)

def main():
    print('Process start')
    config = load_config('process/config.yaml')
    
    communes = config['General']['communes']
    first_day, last_day = resolve_dates(config['Get_weather_data']['first_day'], config['Get_weather_data']['last_day'])
    
    for commune in communes:
        if commune=="Bruxelles-Ville":
            weather_df = get_weather_df("Brussels", first_day, last_day)
        else:
            weather_df = get_weather_df(commune, first_day, last_day)
        try :
            with open(f'data/{commune}/weather.json', 'r', encoding='utf-8') as file:
                old_data = json.load(file)
            old_df = pd.DataFrame(data=old_data['data'], columns=old_data['columns'])
            weather_df = pd.concat([old_df, weather_df])
            weather_df = weather_df.reset_index()
        except :
            os.makedirs(f'data/{commune}', exist_ok=True)
        weather_df.to_json(f'data/{commune}/weather.json', orient='split', date_format='iso')
    print('Process complete')

if __name__ == "__main__":
    """Process for retrieving weather data and adding it to existing data."""
    main()
