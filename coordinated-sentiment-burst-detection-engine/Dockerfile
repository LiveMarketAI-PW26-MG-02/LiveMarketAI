FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY coordinated_sentiment_burst_detection_engine ./coordinated_sentiment_burst_detection_engine
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "coordinated_sentiment_burst_detection_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
