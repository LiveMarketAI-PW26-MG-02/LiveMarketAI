FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai_transparency_layer_for_robo_advisors ./ai_transparency_layer_for_robo_advisors
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "ai_transparency_layer_for_robo_advisors.main:app", "--host", "0.0.0.0", "--port", "8000"]
