import streamlit as st
import numpy as np
import joblib

st.title("Prédiction sur les comptes bancaires")
st.subheader('Application réalisée par Lahad')
st.markdown("cette application utilise le model machine learning pour prédire quelles personnes sont les plus susceptibles d’avoir ou d’utiliser un compte bancaire.")

# Chargement du model
model = joblib.load(filename='model_stream2.joblib')

#Définition d'une fonction
def caracteristique(country, year, uniqueid, location_type, cellphone_access, household_size,
                    age_of_respondent, gender_of_respondent, relationship_with_head, marital_status,
                    education_level, job_type):
    
    
    data = np.array([country, year, uniqueid, location_type, cellphone_access, household_size,
                     age_of_respondent, gender_of_respondent, relationship_with_head, marital_status,
                     education_level, job_type])
    pred = model.predict(data.reshape(1,-1))
    return pred

# Saisissez une valeur pour chaque caracteristique de l'appartement
education_level = st.number_input(label='education_level: ',min_value=0)
job_type = st.number_input(label='job_type: ',min_value=0)
country = st.number_input(label='country: ',min_value=0)
year = st.number_input(label='year: ',min_value=0)
uniqueid = st.number_input(label='uniqueidlocation_type: ',min_value=0)
location_type = st.number_input(label='location_type: ',min_value=0)
cellphone_access= st.number_input(label='cellphone_access : ',min_value=0)
household_size = st.number_input(label='household_size: ',min_value=0)
age_of_respondent = st.number_input(label='age_of_respondent: ',min_value=0)
gender_of_respondent = st.number_input(label='gender_of_respondent: ',min_value=0)
relationship_with_head = st.number_input(label='relationship_with_head: ',min_value=0)
marital_status = st.number_input(label='marital_status: ',min_value=0)


# Création du button 'Predict' qui retourne les prédiction du model
if st.button('Predict'):
    prediction = caracteristique(country, year, uniqueid,location_type,cellphone_access,
                                 household_size, age_of_respondent, gender_of_respondent,
                                 relationship_with_head, marital_status, education_level, job_type)
    
    resultat=prediction[0]
    st.write(resultat)
    
    if resultat == 0:
        st.success("0 sidnifie Cette personne utilise un compte bancaire")
    else: 
        st.warning("1 sidnifie Cette personne n'utilise pas de compte bancaire")
    
    