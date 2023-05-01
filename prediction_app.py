# Importation des librairies
from fastapi import FastAPI
import uvicorn
import pandas as pd
import mlflow
from pydantic import BaseModel

from model import *


# Chargement du modèle MLflow ---> Dans app.py
# model = mlflow.sklearn.load_model("C:\\Users\\lea\\Documents\\PROJET7\\mlflow_model")


# Création de l'application FastAPI
app = FastAPI()

class predirectionRequestObject(BaseModel):
    client_id: float

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/name')
def get_name():
    name_2 = "test"
    return {'message': name_2}


# Définition du endpoint de prédiction pour la décision d'octroi de crédit
# Expose the prediction functionality, make a prediction from the passed
# JSON data and return the predicted decision the confidence
@app.post('/predict_credit_decision')
async def predict_credit_decision(data: predirectionRequestObject):
    client_id = data.client_id
    proba, prediction = predict(client_id)
    return {"result" : prediction, "proba" : proba}

    # client_data_filtered = clients_data.loc[clients_data['SK_ID_CURR']==client_id]
    # sexe = client_data_filtered["CODE_GENDER"].values[0]
    # Retourner la prédiction dans la réponse avec l'identifiant du client
    # return {"sexe": sexe}

# Définition du endpoint de retour de la liste des clients
@app.post('/get_clients_list')
async def get_clients_list():
    return {"clients_list" : clients_id_list()}

# Définition du endpoint pour recuperer les informations sur un client
@app.post('/get_client_data')
async def get_client_data(data: predirectionRequestObject):
    return {"client_data" : client_info(data.client_id)}

# Définition du endpoint pour recuperer les informations du crédit
@app.post('/get_credit_info')
async def get_credit_info(data: predirectionRequestObject):
    return {"credit_info" : credit_info(data.client_id)}

   