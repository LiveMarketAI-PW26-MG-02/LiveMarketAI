FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY interpretable_crypto_whale_influence_mapper ./interpretable_crypto_whale_influence_mapper
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "interpretable_crypto_whale_influence_mapper.main:app", "--host", "0.0.0.0", "--port", "8000"]
