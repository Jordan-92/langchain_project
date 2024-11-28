# AI Project: Air Quality Prediction

## 1. Configuration
- Clone the git repository.
- Create a python environment and install the requirements.
- Create a dotenv file and put the apis in it.
    ```env
    GOOGLE_API_KEY=
    GOOGLE_CSE_ID=
    OPENAI_API_KEY=
    WEATHER_API_KEY=
    ```
- Start the api by "python API".
- Start the application using by example the extension "Live Server".

## 2. Retrieve Data Process
There is already data to try the application with “Anderlecht” and “Bruxelles-Ville”.

### 2.1 Configure congif.yaml file
Customize the config.yaml file to set up parameters for data retrieval:
```yaml
Enable: True    # Set to “True” to enable the process
General:
  communes: ["Anderlecht"]   # Replace or add more locations if needed
Get_weather_data:
  first_day: "2024-11-22"    # Start date for weather data retrieval (or "Yesterday")
  last_day: "Max"            # End date for weather data retrieval (or "Max")
```

### 2.2 Get historical pm10 and pm2.5 data
Run the pm10/pm2.5 data retrieval script "get_historical_pm10_pm25.py" to fetch historical pm10 and pm2.5 data for the configured communes.

### 2.3 Get weather data
Run the weather data retrieval script "get_weather_data.py" to fetch historical or/and forecasted weather data for the configured communes.


#   l a n g c h a i n _ p r o j e c t  
 