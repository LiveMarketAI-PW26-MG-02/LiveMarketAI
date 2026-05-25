FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY deepfake_ceo_speech_manipulation_detector ./deepfake_ceo_speech_manipulation_detector
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "deepfake_ceo_speech_manipulation_detector.main:app", "--host", "0.0.0.0", "--port", "8000"]
