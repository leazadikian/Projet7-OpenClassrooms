import pandas as pd
import streamlit as st
import requests
import json
import time
from PIL import Image
import base64

# Chargement du logo Prêt à Dépenser
<<<<<<< HEAD
local_path=''
heroku_path='Dashboard/'
path=heroku_path

logo_image = Image.open(path + 'logo_pret_a_depenser.png')
=======
logo_image = Image.open('Dashboard/logo_pret_a_depenser.png')
>>>>>>> a670054891ee72bf8b5e0a1b0a19807d02997849



#############################
# FONCTIONS REQUÊTE A L'API #
#############################

base_url = "http://127.0.0.1:8000/"
headers_request = {"Content-Type": "application/json"}

@st.cache_data
def request_prediction(client_id):
    url_request = base_url + "predict_credit_decision"
    data_json = {"client_id" : client_id}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
        
    return response.json()["proba"], response.json()["result"]

@st.cache_data
def request_client_data(client_id):
    url_request = base_url + "get_client_data"
    data_json = {"client_id" : client_id}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
        
    return pd.DataFrame.from_dict(response.json()["client_data"])

@st.cache_data
def request_credit_info(client_id):
    url_request = base_url + "get_credit_info"
    data_json = {"client_id" : client_id}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
        
    return pd.DataFrame.from_dict(response.json()["credit_info"])

@st.cache_data    
def request_client_list():
    url_request = base_url + "get_clients_list"
    response = requests.request( method='POST', headers=headers_request, url=url_request)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return [int(x) for x in response.json()["clients_list"]]

@st.cache_data
def request_feature_definition():
    url_request = base_url + "get_features_definition"
    response = requests.request( method='POST', headers=headers_request, url=url_request)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return response.json()["feature_definition"]

@st.cache_data
def request_shap_waterfall_chart(client_id, feat_number):
    url_request = base_url + "get_shap_waterfall_chart"
    data_json = {"client_id" : client_id, "feat_number" : feat_number}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return response.json()["base64_image"]

@st.cache_data
def request_shap_waterfall_chart_global(feat_number):
    url_request = base_url + "get_shap_waterfall_chart_global"
    data_json = {"feat_number" : feat_number}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return response.json()["base64_image"]

@st.cache_data
def request_comparison_chart(client_id, feat_name):
    url_request = base_url + "get_comparison_graph"
    data_json = {"client_id" : client_id, "feat_name" : feat_name}
    response = requests.request( method='POST', headers=headers_request, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return response.json()["base64_image"]


########################
# FONCTION PRINCIPALE #
########################
def main():
    
    st.set_page_config(page_title='Prêt à Dépenser - Léa Zadikian', page_icon  = logo_image, layout = 'wide', initial_sidebar_state = 'auto')
    
    #########
    # TITRE #
    #########
    st.header("PRÊT À DÉPENSER")
    st.markdown("<h1 style='text-align: center; border: 2px solid black; padding: 10px; background-color: #cccccc; border-radius: 10px;'> Outil d'aide à la décision d'octroi d'un crédit</h1>", unsafe_allow_html=True)
    

    ############
    # SIDEBAR #
    ###########
    
    # Affichage du logo de l'entreprise
    st.sidebar.image(logo_image,use_column_width=True)
    
    # Sélection d'un client par l'utilisateur
    id_input = st.sidebar.selectbox("**Sélectionner l'identifiant du client :**", request_client_list())
    
    # Affichage des données personnelles du client sélectionné
    if id_input:
        st.sidebar.subheader('Informations générales du client %s : ' % id_input)  
        st.sidebar.write(request_client_data(id_input))
    
    # Caractéristiques du prêt demandé par le client sélectionné 
    if id_input:
        st.sidebar.subheader('Caractéristiques du prêt demandé : ')
        st.sidebar.write(request_credit_info(id_input))     
    
    
   ################### 
   # Page principale #
   ###################
    
    st.write(" ") # ligne vide pour laisser un espace
    st.write(" ") # ligne vide pour laisser un espace

    st.subheader('Client n° %s' % id_input)  
    
    st.write(" ") # ligne vide pour laisser un espace
    st.write(" ") # ligne vide pour laisser un espace
    
    # 1 # Affichage de la prédiction de la solvabilité du client sélectionné       
    if st.checkbox('**Prédire la solvabilité du client**'):
        
        # On demande à l'API de la prédiction de classe et de la probabilité pour le client sélectionné
        proba, prediction = request_prediction(id_input) # prédiction de probabilité et de la classe 
      
        st.markdown("* Probabilité de rembousement du client : **%0.2f %%**" % (proba*100))
        
        # Barre de progression affichant le % de chance de remboursement du client (et non pas le % de risque)
        progress_bar = st.progress(0)
        for i in range(round(proba*100)):
            progress_bar.progress(i + 1)
                
        st.markdown("* La réponse suggérée pour la demande de prêt du client est : ")
        # Si la prediction vaut 1, on affiche "crédit refusé" sur bandeau rouge, 
        # si prediction vaut 0, on affiche "crédit accordé" sur bandeau vert
        if prediction == 1:
            st.error("Crédit refusé !")
        elif prediction == 0:
            st.success("Crédit accordé !")
            
        
     # 2 # Affichage des features importance locale et globale (explication de la prédiction)    
    if st.checkbox("**Afficher l'explication de la prédiction**"):
      
        # Sélection par l'utilisateur du nombre de features à afficher
        feat_number = st.slider("* Sélectionner le nombre de paramètres souhaité pour expliquer la prédiction", 1, 30, 10)
        
        col1, col2 = st.columns(2)
        
        col1.header("Client")
        base64_image = request_shap_waterfall_chart(id_input,feat_number)
        image = base64.b64decode(base64_image)
        col1.image(image)
            
        col2.header("Global")
        base64_image = request_shap_waterfall_chart_global(feat_number)
        image = base64.b64decode(base64_image)
        col2.image(image)
        
  
    # 3 # Comparaison avec les autres clients
    if st.checkbox('**Comparer le client avec les autres clients**'):
        
        feature_name = st.selectbox('Sélectionner un paramètre :', [
                "AMT_INCOME_TOTAL",
                "DAYS_EMPLOYED",
                "REGION_POPULATION_RELATIVE",
                "DAYS_BIRTH",
                "AMT_CREDIT"])
        
        with st.spinner('Chargement du graphique en cours...'):
            base64_image = request_comparison_chart(id_input,feature_name)
            image = base64.b64decode(base64_image)
            st.image(image)
            
           
    # 4 # Définition des features
    if st.checkbox("Voir la définition des paramètres") :
        df_features = pd.DataFrame.from_dict(request_feature_definition())
        features_name = sorted(df_features.index.unique().to_list())

        feature = st.selectbox('Selectionner un paramètres…', features_name)
        st.table(df_features.loc[df_features.index == feature][:1])

if __name__ == '__main__':
    main()