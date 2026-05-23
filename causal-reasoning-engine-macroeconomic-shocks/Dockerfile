FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY causal_reasoning_engine_macroeconomic_shocks ./causal_reasoning_engine_macroeconomic_shocks
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "causal_reasoning_engine_macroeconomic_shocks.main:app", "--host", "0.0.0.0", "--port", "8000"]
