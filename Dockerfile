FROM python:3.11-slim

WORKDIR /app

RUN pip install Flask flask-cors scikit-learn pandas joblib
COPY modelos.pkl .
COPY controller.py .

EXPOSE 8080

CMD ["python", "controller.py"]
