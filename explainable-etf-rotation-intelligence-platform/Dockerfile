FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY explainable_etf_rotation_intelligence_platform ./explainable_etf_rotation_intelligence_platform
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "explainable_etf_rotation_intelligence_platform.main:app", "--host", "0.0.0.0", "--port", "8000"]
