from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from Utils import utils as u


def plot_loss(history, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Curve for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

df = u.merge_pm_and_weather('Anderlecht', 'Clos Mudra')
df_filtred = u.filter(df)

X = df_filtred[['temp', 'humidity', 'precip', 'snow', 'windspeed', 'pressure', 'uvindex', 'moonphase']].values
y = df_filtred[['pm10', 'pm25']].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

results = {}
models = {
    'Dense Model': u.create_dense_model(X_train.shape[1]),
    'LSTM Model': u.create_lstm_model(X_train.shape[1])
}

for name, model in models.items():
    if 'LSTM' in name:
        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        history = model.fit(X_train_reshaped, y_train, validation_data=(X_test_reshaped, y_test), epochs=12, batch_size=32, verbose=0)
        y_pred = scaler_y.inverse_transform(model.predict(X_test_reshaped))
    else:
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12, batch_size=32, verbose=0)
        y_pred = scaler_y.inverse_transform(model.predict(X_test))

    plot_loss(history, name)

    y_test_original = scaler_y.inverse_transform(y_test)
    mae = mean_absolute_error(y_test_original, y_pred)
    mse = mean_squared_error(y_test_original, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse}
    print(f"{name} - MAE: {mae:.2f}, MSE: {mse:.2f}")

results_df = pd.DataFrame(results).T
print(results_df)