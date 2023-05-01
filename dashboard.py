import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import requests
import json
import time

from model import *

# Chargement du logo Prêt à Dépenser
logo_image = Image.open('logo_pret_a_depenser.png')


#############################
# FONCTIONS REQUÊTE A L'API #
#############################
def request_prediction(client_id):
    url_request = "http://127.0.0.1:8000/predict_credit_decision"
    headers = {"Content-Type": "application/json"}
    data_json = {"client_id" : client_id}
    response = requests.request( method='POST', headers=headers, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
        
    return response.json()["proba"], response.json()["result"]

def request_client_data(client_id):
    url_request = "http://127.0.0.1:8000/get_client_data"
    headers = {"Content-Type": "application/json"}
    data_json = {"client_id" : client_id}
    response = requests.request( method='POST', headers=headers, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
        
    return pd.DataFrame.from_dict(response.json()["client_data"])

def request_credit_info(client_id):
    url_request = "http://127.0.0.1:8000/get_credit_info"
    headers = {"Content-Type": "application/json"}
    data_json = {"client_id" : client_id}
    response = requests.request( method='POST', headers=headers, url=url_request, json=data_json)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
        
    return pd.DataFrame.from_dict(response.json()["credit_info"])
    
def request_client_list():
    url_request = "http://127.0.0.1:8000/get_clients_list"
    #headers = {"Content-Type": "application/json"}
    #data_json = {"client_id" : client_id}
    response = requests.request( method='POST', url=url_request)
    
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    
    return response.json()["clients_list"]




########################
# FONCTION PRINCIPALE #
########################
def main():
    PREDICTION_API = 'http://127.0.0.1:5000'
    
    
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
            
 
        
     # 2 # Affichage des features importance locale (explication de la prédiction)    
    if st.checkbox("**Afficher l'explication de la prédiction**"):
      
        # Sélection par l'utilisateur du nombre de features à afficher
        feat_number = st.slider("* Sélectionner le nombre de paramètres souhaité pour expliquer la prédiction", 0, 30, 10)
        
        with st.spinner('Chargement du graphique en cours...'):
            # Affichage du graph summary_plot de shap ( explication d'une prédiction individuelle)
            st.pyplot(prediction_SHAP_summary_plot(id_input,feat_number))
        
        with st.spinner('Chargement du graphique en cours...'):
            # Affichage du graph watefall de shap ( explication d'une prédiction individuelle)
            st.pyplot(prediction_SHAP_waterfall(id_input,feat_number))

           
    
    # 3 # Comparaison avec les autres clients
    if st.checkbox('**Comparer le client avec les autres clients**'):
        
        feature_name = st.selectbox('Sélectionner un paramètre :', [
                "AMT_INCOME_TOTAL",
                "DAYS_EMPLOYED",
                "REGION_POPULATION_RELATIVE",
                "DAYS_BIRTH",
                "AMT_CREDIT"])
        
        with st.spinner('Chargement du graphique en cours...'):
            fig=comparison_graph(id_input,feature_name)
            st.pyplot(fig,use_container_width=True)

            
            #fig, ax = plt.subplots(feature_name)
            #plt.title('Distribution du paramètre %s ' % feature_name)
            #sns.histplot( data=prod_data,x=feature_name)
            #ax.axvline(int(infos_client["feature_name"].values / 365), color="green", linestyle='--')
        
 
        
        
    # 4 # Définition des features
    if st.checkbox("Voir la définition des paramètres") :
        feature = st.selectbox('Selectionner un paramètres…', features_def())
        st.table(description.loc[description.index == feature][:1])


if __name__ == '__main__':
    main()
