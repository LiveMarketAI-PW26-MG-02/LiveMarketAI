FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY discord_meme_stock_manipulation_tracker ./discord_meme_stock_manipulation_tracker
COPY pytest.ini .
EXPOSE 8000
CMD ["uvicorn", "discord_meme_stock_manipulation_tracker.main:app", "--host", "0.0.0.0", "--port", "8000"]
