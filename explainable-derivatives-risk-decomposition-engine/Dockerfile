FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY explainable_derivatives_risk_decomposition_engine ./explainable_derivatives_risk_decomposition_engine
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "explainable_derivatives_risk_decomposition_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
