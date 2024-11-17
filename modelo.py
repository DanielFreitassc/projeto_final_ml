import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('fuel.csv')

df['ENGINE SIZE'] = pd.to_numeric(df['ENGINE SIZE'], errors='coerce')
df['CYLINDERS'] = pd.to_numeric(df['CYLINDERS'], errors='coerce')

df['FUEL'] = df['FUEL'].map({'X': 0, 'Z': 1, 'E': 2, 'N': 3, 'D': 4})

df.dropna(subset=['ENGINE SIZE', 'CYLINDERS', 'FUEL', 'COMB (L/100 km)'], inplace=True)

y = 100 / df['COMB (L/100 km)']
X = df[['ENGINE SIZE', 'CYLINDERS', 'FUEL']]

preprocessador = ColumnTransformer(
    transformers=[('num', Pipeline([
                          ('scaler', StandardScaler()),
                          ('passthrough', 'passthrough')
                      ]), ['ENGINE SIZE', 'CYLINDERS']),
                  ('cat', 'passthrough', ['FUEL'])]
)

modelos = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=150, min_samples_split=2, max_depth=30, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=7, learning_rate=0.2, random_state=42)
}

pipelines = {
    nome: Pipeline(steps=[('preprocessador', preprocessador), ('modelo', modelo)])
    for nome, modelo in modelos.items()
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

melhor_modelo_nome = None
melhor_modelo_pipeline = None
melhor_r2 = float('-inf')

metricas = {}
for nome, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train) 
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)  
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    metricas[nome] = {'R²': r2, 'RMSE': rmse} 

    if r2 > melhor_r2: 
        melhor_r2 = r2
        melhor_modelo_nome = nome
        melhor_modelo_pipeline = pipeline

joblib.dump(melhor_modelo_pipeline, 'modelos.pkl')

print(f'Melhor modelo: {melhor_modelo_nome}')
print(f'Métricas: {metricas}')
