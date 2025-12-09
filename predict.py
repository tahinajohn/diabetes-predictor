# from data_enum import PatientData
# from train import train_model
# from model import diabetes_model
# from processing import X, y

# def predict(data: dict):
#     """
#     Predict the outcome for a given patient's data.

#     Args:
#         data (PatientData): The patient's data.
#     """
#     print("-----------------------------------------")
#     #print("Model trained for prediction....")
#     #print("pregnancies:", data.pregnancies)
#     model = diabetes_model()
#     # train_model(model)
#     model.fit(X, y)
#     prediction = model.predict(data)
#     return prediction[0]