FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cross_exchange_spoofing_detection_ai ./cross_exchange_spoofing_detection_ai
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "cross_exchange_spoofing_detection_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
