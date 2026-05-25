FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY emotion_driven_volatility_prediction_model ./emotion_driven_volatility_prediction_model
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "emotion_driven_volatility_prediction_model.main:app", "--host", "0.0.0.0", "--port", "8000"]
