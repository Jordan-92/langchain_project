import os
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from Utils import utils as u


def predict(commune, station):
    df = u.merge_pm_and_weather(commune, station)
    df_filtred = u.filter(df)

    X = df_filtred[['year', 'month', 'day_of_week', 'temp', 'humidity', 'precip', 'snow', 'windspeed', 'pressure', 'uvindex', 'moonphase']].values
    y = df_filtred[['pm10', 'pm25']].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    model = u.create_lstm_model(X_train.shape[1])
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=12, batch_size=32, verbose=0)
    station = station.replace(" ", "_")

    with open(f'data/{commune}/{station}.pkl', 'wb') as file:
        pickle.dump({
            'model': model,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }, file)

    print("Modèle et scalers enregistrés dans 'model.pkl'.")

def main():
    config = u.load_config('process/config.yaml')
    stations = u.get_content_from_yaml('communes.yaml')
    communes = config['General']['communes']
    for commune in communes:
        for id, station in stations[commune].items():
            predict(commune, station)

if __name__ == "__main__":
    main()
