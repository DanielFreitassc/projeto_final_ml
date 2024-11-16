import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

@st.cache_data
def carregar_dados():
    return pd.read_csv('fuel.csv')

df = carregar_dados()

y = df['COMB (L/100 km)']
X = df[['ENGINE SIZE', 'CYLINDERS', 'TRANSMISSION', 'FUEL']]

df['FUEL'] = df['FUEL'].map({
    'X': 0, 'Z': 1, 'E': 2, 'N': 3, 'D': 4
})

preprocessador = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['TRANSMISSION', 'FUEL']),
        ('num', StandardScaler(), ['ENGINE SIZE', 'CYLINDERS'])
    ])

rf_best_model = RandomForestRegressor(
    n_estimators=100,
    min_samples_split=5,
    max_depth=30,
    random_state=42
)

gb_best_model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=7,
    learning_rate=0.1,
    random_state=42
)

pipelines = {
    'Linear Regression': Pipeline(steps=[
        ('preprocessador', preprocessador),
        ('modelo', LinearRegression())
    ]),
    'Random Forest': Pipeline(steps=[
        ('preprocessador', preprocessador),
        ('modelo', rf_best_model)
    ]),
    'Gradient Boosting': Pipeline(steps=[
        ('preprocessador', preprocessador),
        ('modelo', gb_best_model)
    ])
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for modelo in pipelines.values():
    modelo.fit(X_train, y_train)

def prever_consumo(engine_size, cylinders, transmission, fuel, modelo_pipeline):
    novos_dados = pd.DataFrame({
        'ENGINE SIZE': [engine_size],
        'CYLINDERS': [cylinders],
        'TRANSMISSION': [transmission],
        'FUEL': [fuel]
    })
    consumo_previsto = modelo_pipeline.predict(novos_dados)
    consumo_por_km = 100 / consumo_previsto[0] 
    return consumo_por_km

st.title('Previsão de Consumo de Combustível')

engine_size = st.number_input('Tamanho do Motor (em litros)', min_value=0.0, max_value=10.0, step=0.1)
cylinders = st.selectbox('Número de Cilindros', [4, 6, 8])
transmission = st.selectbox('Transmissão', ['A4', 'M5', 'A5', 'AS5', 'M6'])

fuel_type = st.selectbox('Tipo de Combustível', ['Gasolina Comum', 'Gasolina Premium', 'Etanol', 'Gás Natural', 'Diesel'])

fuel_mapping = {
    'Gasolina Comum': 'X',    
    'Gasolina Premium': 'Z',  
    'Etanol': 'E',            
    'Gás Natural': 'N',        
    'Diesel': 'D'              
}

fuel = fuel_mapping.get(fuel_type)

if st.button('Prever Consumo'):
    consumo_rf = prever_consumo(engine_size, cylinders, transmission, fuel, pipelines['Random Forest'])
    st.subheader('Random Forest')
    st.write(f'Consumo previsto: {consumo_rf:.2f} km/L')

    consumo_gb = prever_consumo(engine_size, cylinders, transmission, fuel, pipelines['Gradient Boosting'])
    st.subheader('Gradient Boosting')
    st.write(f'Consumo previsto: {consumo_gb:.2f} km/L')

    modelo_lr = pipelines['Linear Regression']
    modelo_lr.fit(X_train, y_train)
    consumo_lr = prever_consumo(engine_size, cylinders, transmission, fuel, modelo_lr)
    st.subheader('Linear Regression')
    st.write(f'Consumo previsto: {consumo_lr:.2f} km/L')
