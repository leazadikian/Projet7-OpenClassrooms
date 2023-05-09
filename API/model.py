# Importation des librairies
import pandas as pd
import mlflow
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import base64

###################################################
#     FONCTION DE TRANSFORMATION DES DONNEES      #
###################################################

# Les transformations appliquées aux données d'entrée
def transform(df):
    return pd.read_csv("test_df_imputed.csv")

###################################################
#       CHARGEMENT DES DONNEES ET DU MODELE       #
###################################################

# Chargement des données des clients depuis un fichier CSV
prod_data = data = pd.read_csv("application_test.csv") # base de clients production
clients_data = transform(prod_data)
# Pour test, data client de test déjà transformées. Pour finaliser le projet il faudra supprimer cet import de données. Ce ddf sera obtenu par transformation de "prod_data" 

feats = [f for f in clients_data.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index','Unnamed: 0']]
#client_data_wo_id = clients_data[feats]

# Chargement depuis un fichier csv d'un dataframe contenant la description des colonnes
colmumn_description_df = pd.read_csv("HomeCredit_columns_description.csv", usecols=['Row', 'Description'], index_col=0, encoding= 'unicode_escape')

# Chargement du modèle MLflow
#logged_model = 'runs:/b8b6c9ae221242408a65c79dd1f22f11/model'
#loaded_model = mlflow.pyfunc.load_model(logged_model)
#loaded_model = mlflow.xgboost.load_model(logged_model)
loaded_model = pickle.load(open("model.pck","rb"))

###################################################
# FONCTION D'INFORMATION GENERALES SUR LE DATASET #
###################################################

# Retourne un dict contenant la description des colonnes
def features_def():
    return colmumn_description_df.to_dict()

# Retourne la liste des identifiants clients de la base de données
def clients_id_list():
     #return prod_data['SK_ID_CURR'].tolist() # pour le projet final
    return sorted(clients_data['SK_ID_CURR'].tolist()) # pour les tests


##################################################################
# FONCTION DE PREDICTION ET D'EXTRACTION DES DONNEES D'UN CLIENT #
##################################################################

# Retourne les informations personnelles sur le client
def client_info(client_id):
    client_info_columns = [
                 "CODE_GENDER",
                 "CNT_CHILDREN",
                 "FLAG_OWN_CAR",
                 "FLAG_OWN_REALTY",
                 "NAME_FAMILY_STATUS",
                 "NAME_HOUSING_TYPE",
                 "NAME_EDUCATION_TYPE",
                 "NAME_INCOME_TYPE",
                 "OCCUPATION_TYPE",
                 "AMT_INCOME_TOTAL"
                 ]    
    client_info=prod_data.loc[prod_data['SK_ID_CURR']==client_id,client_info_columns].T # informations client pour le client selectionné
    client_info= client_info.fillna('N/A')
    return client_info

# Retourne les saractéristiques du crédit demandé par le client
def credit_info(client_id):
    credit_info_columns=["NAME_CONTRACT_TYPE","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE","REGION_POPULATION_RELATIVE"]
    credit_info=prod_data.loc[prod_data['SK_ID_CURR']==client_id,credit_info_columns].T # informations crédit pour le client selectionné
    return credit_info

# Retourne la prediction sur le client
def predict (client_id):
    selected_client=clients_data.loc[clients_data['SK_ID_CURR']==client_id]
        
    prediction_proba = loaded_model.predict_proba(selected_client[feats]).tolist()[0][0]
    prediction = loaded_model.predict(selected_client[feats]).tolist()[0]
    
    return prediction_proba, prediction

#########################################
# FONCTION DE GENERATION DES GRAPHIQUES #
#########################################

# Retourne une image du graphique SHAP waterfall pour un client donné et un nombre de feature donné
def shap_waterfall_chart(client_id,feat_number):
    
    index_selected=clients_data.loc[clients_data['SK_ID_CURR']==client_id].index[0] # index correspondant au client sélectionné

    fig, ax = plt.subplots()
    plt.title("Importance des paramètres dans la décision d'octroi ou de refus")
    explainer = shap.Explainer(loaded_model, clients_data)
    shap_values = explainer(clients_data)    # compute SHAP values
    shap.plots.waterfall(shap_values[index_selected], max_display=feat_number, show=False)
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image = base64.b64encode(buffer.getvalue())
                             
    return image

# Retourne une image du graphique SHAP waterfall global pour un nombre de feature donné
def shap_waterfall_chart_global(feat_number):

    fig, ax = plt.subplots()
    plt.title("Importance globale des paramètres")
    explainer = shap.Explainer(loaded_model, clients_data)
    shap_values = explainer(clients_data)    # compute SHAP values
    
    shap_values.values = shap_values.values.mean(axis=0)
    shap_values.base_values = shap_values.base_values.mean()
    shap_values.data=shap_values.data.mean(axis=0)
    
    shap.plots.waterfall(shap_values, max_display=feat_number , show=False)
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image = base64.b64encode(buffer.getvalue())
                             
    return image

# Graphique de comparaison des informations descriptives du client par rapport à l'ensmeble des clients
def comparison_graph(client_id, feature_name):
    
    selected_client=clients_data.loc[clients_data['SK_ID_CURR']==client_id]
    fig, ax = plt.subplots()
    mean = clients_data[feature_name].mean()
    std=clients_data[feature_name].std()
    sns.histplot(prod_data,x=feature_name, color='grey')  
    ax.axvline(int(selected_client[feature_name]), color="blue", linestyle='--',linewidth=2, label ='Client n° %s'%client_id)
    ax.axvline(int(mean), color="black", linestyle='--',linewidth=2, label ='Moyenne des clients : %d'%mean)
    ax.set(title='Distribution du paramètre %s' % feature_name, ylabel='')
    #ax.set_xlim(mean - 5 * std, mean + 5 * std)
    plt.grid(axis='y')
    plt.legend()
    
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image = base64.b64encode(buffer.getvalue())
    
    return image