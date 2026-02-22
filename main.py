from typing import Union
from fastapi import FastAPI
from bonus import bonus
from total import total
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}



@app.get("/bonus/{basic_salary}")
def calculate_salary(basic_salary: float):
    bonus_amount = bonus(basic_salary)
    return {"basic_salary": basic_salary, "bonus": bonus_amount, "total": basic_salary + bonus_amount}
