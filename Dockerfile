FROM python:3.11-slim

WORKDIR /usr/src/app

RUN pip install Flask flask-cors scikit-learn pandas joblib

COPY . .

EXPOSE 8080

CMD ["python", "modelo_final.py"]
