import pickle
from langchain.agents import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
import json
from dotenv import load_dotenv
import pandas as pd
from Utils import utils as u
load_dotenv()

COMMUNE = "Anderlecht"
STATION = "Clos_Mudra"

search = TavilySearchResults()

def predict_air_quality(date):
    try:
        with open(f"data/{COMMUNE}/weather.json", "r", encoding='utf-8') as f:
            weather_data = json.load(f)

        df = pd.DataFrame(data=weather_data["data"], columns=weather_data["columns"])

        df = df[df['datetime'] == date]
        if df.empty:
            return f"Aucune donnée météo disponible pour la date {date}."

        df = u.filter(df)
        data_array = df.values

        with open(f"data/{COMMUNE}/{STATION}.pkl", "rb") as model_file:
            model_data = pickle.load(model_file)

        model = model_data["model"]
        scaler_X = model_data["scaler_X"]
        scaler_y = model_data["scaler_y"]


        scaled_data = scaler_X.transform(data_array)

        scaled_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

        predictions_scaled = model.predict(scaled_data)
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        return predictions.tolist()
    except Exception as e:
        return f"Erreur lors de la prédiction : {str(e)}"


tools = [
    Tool(
        name="google_search",
        func=search.run,
        description="Useful for answering questions about future projects."
    ),
    Tool(
        name="air_quality_prediction",
        func=predict_air_quality,
        description="Useful for answering questions about the air quality on a date that need to be precised. Only the date need to be precised and the output is gonna be pm10 and pm2.5"
    )
]