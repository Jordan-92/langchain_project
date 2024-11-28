from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from Utils import utils as u

df = u.merge_pm_and_weather('Anderlecht', 'Clos Mudra')
df_filtred = u.filter(df)

X = df_filtred[['temp', 'humidity', 'precip', 'snow', 'windspeed', 'pressure', 'uvindex', 'moonphase']]
y = df_filtred[['pm10', 'pm25']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Support Vector Regressor (SVR)": SVR()
}

model_performances = {}

for model_name, model in models.items():
    if model_name == "Support Vector Regressor (SVR)":
        model.fit(X_train_scaled, y_train['pm10'])
    else:
        model.fit(X_train, y_train['pm10'])

    if model_name == "Support Vector Regressor (SVR)":
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test['pm10'], y_pred)
    r2 = r2_score(y_test['pm10'], y_pred)

    model_performances[model_name] = {"MSE": mse, "R2": r2}

print("\nRésumé des performances des modèles pour PM10:")
for model_name, metrics in model_performances.items():
    print(f"{model_name} - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")