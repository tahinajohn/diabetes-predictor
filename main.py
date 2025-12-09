from fastapi import FastAPI

# from data_enum import PatientData
# from predict import predict
import joblib

from data_enum import PatientData

model = joblib.load("diabetes_model.pkl")
import numpy as np
app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
async def make_prediction(data : PatientData):
    pregnancies = data.pregnancies
    glucose = data.glucose
    blood_pressure = data.blood_pressure
    skin_thickness = data.skin_thickness
    insulin = data.insulin
    bmi = data.bmi
    diabetes_pedigree_function = data.diabetes_pedigree_function
    age = data.age

    #prediction = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])
    test_pred = model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    return {"prediction " : int(test_pred[0])}

@app.post("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello, {name}!"}

@app.get("/status/")
async def get_status():
    return {"status": "API is running"}
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="localhost", port=8000, reload=True)