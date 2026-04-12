from fastapi import FastAPI
from app import models, crud

app = FastAPI()

models.create_table()

@app.get("/")
def home():
    return {"message": "Holdings API Running"}


@app.post("/add")
def add(instrument_id: str, quantity: int):
    crud.add_holding(instrument_id, quantity)
    return {"message": "Added successfully"}


@app.get("/holdings")
def holdings():
    return crud.get_holdings()


@app.get("/total")
def total():
    return {"total_value": crud.total_value()}


@app.get("/summary")
def get_summary():
    return crud.summary()