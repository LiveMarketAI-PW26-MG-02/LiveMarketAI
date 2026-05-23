FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dark_pool_manipulation_intelligence_engine ./dark_pool_manipulation_intelligence_engine
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "dark_pool_manipulation_intelligence_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
