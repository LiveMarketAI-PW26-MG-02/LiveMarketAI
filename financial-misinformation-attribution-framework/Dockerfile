FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY financial_misinformation_attribution_framework ./financial_misinformation_attribution_framework
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "financial_misinformation_attribution_framework.main:app", "--host", "0.0.0.0", "--port", "8000"]
