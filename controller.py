from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

pipelines = joblib.load('modelos.pkl')

def prever_consumo(engine_size, cylinders, fuel, modelo_pipeline):
    if any(pd.isnull(val) for val in [engine_size, cylinders, fuel]):
        raise ValueError("Os dados de entrada não podem conter valores nulos")

    novos_dados = pd.DataFrame({
        'ENGINE SIZE': [engine_size],
        'CYLINDERS': [cylinders],
        'FUEL': [fuel]
    })
    consumo_previsto = modelo_pipeline.predict(novos_dados)
    return round(consumo_previsto[0], 2) 

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    required_fields = ['engine_size', 'cylinders', 'fuel']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Todos os campos são obrigatórios!"}), 400

    fuel_mapping = {
        'Gasolina Comum': 0,    
        'Gasolina Premium': 1,  
        'Etanol': 2,            
        'Gás Natural': 3,        
        'Diesel': 4              
    }
    
    fuel = fuel_mapping.get(data['fuel'])
    if fuel is None:
        return jsonify({"error": "Tipo de combustível inválido!"}), 400

    engine_size = data['engine_size']
    cylinders = data['cylinders']

    resultados = {}
    if isinstance(pipelines, dict):
        for nome, pipeline in pipelines.items():
            try:
                consumo = prever_consumo(engine_size, cylinders, fuel, pipeline)
                resultados[nome] = f"{consumo} km/L"
            except Exception as e:
                app.logger.error(f"Error processing {nome}: {str(e)}")
                return jsonify({"error": f"Erro ao processar o modelo {nome}: {str(e)}"}), 500
    else:
        try:
            consumo = prever_consumo(engine_size, cylinders, fuel, pipelines)
            return jsonify({"Faz": f"{consumo} km/L"})
        except Exception as e:
            app.logger.error(f"Error processing model: {str(e)}")
            return jsonify({"error": f"Erro ao processar o modelo: {str(e)}"}), 500

    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
