FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai_driven_retail_investor_confidence_meter ./ai_driven_retail_investor_confidence_meter
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "ai_driven_retail_investor_confidence_meter.main:app", "--host", "0.0.0.0", "--port", "8000"]
