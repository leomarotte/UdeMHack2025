import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from predict_model import predict_single_image
from predict_model_parkinson import predict_single_image_parkinson
from predict_model_dementia import predict_single_image_dementia

import csv

# Créer une classe Patient pour enregistrer les données des patients après les tests
class Patient:

     def __init__(self, file_number, cancer, cancer_percentage, dementia, dementia_percentage, parkinson, parkinson_percentage):
          self.file_number = file_number
          self.cancer = cancer
          self.cancer_percentage = cancer_percentage
          self.dementia = dementia
          self.dementia_percentage = dementia_percentage
          self.parkinson = parkinson
          self.parkinson_percentage = parkinson_percentage
     
     def to_dict(self):
        return {
            "File number": self.file_number,
            "Cancer": self.cancer,
            "Cancer percentage": self.cancer_percentage,
            "Dementia": self.dementia,
            "Dementia percentage": self.dementia_percentage,
            "Parkinson" : self.parkinson,
            "Parkinson percentage" : self.parkinson_percentage,
        }

# Charger les deux modèles utilisés
model = tf.keras.models.load_model("modele_cancer.keras")
model_dementia = tf.keras.models.load_model("modele_demence.keras")
model_parkinson = tf.keras.models.load_model("modele_parkinson.keras")

# Centré le titre
style_heading = 'text-align: center'
st.markdown(f"<h1 style='{style_heading}'>Brain abnormalities detection</h1>", unsafe_allow_html=True)

# Identifier les patients par leur numéro de dossier unique
user_input = st.text_input("Enter file number : ")


# Upload l'image du MRI
if user_input is not None :
     upload_file = st.file_uploader (
          "Choose a file:",
          type=["png", "jpg", "jpeg"],
          accept_multiple_files = False   # upload une image à la fois
     )



if upload_file is not None :
    
    # Prédiction de la possbilité de cancer
    prediction = predict_single_image(model, upload_file)
    resultat_cancer = prediction[0]
    pourcentage_cancer = prediction[1]

    # Prédiction de la possibilité de demence
    prediction_2 = predict_single_image_dementia(model_dementia, upload_file)
    resultat_dementia = prediction_2[0]
    pourcentage_dementia = prediction_2[1]
    
    # Prédiction de la possibilité de parkinson  
    prediction_3 = predict_single_image_parkinson(model_parkinson, upload_file)
    resultat_parkinson = prediction_3[0]
    pourcentage_parkinson = prediction_3[1]

    # Afficher le mri du patient
    image = Image.open(upload_file)  
    st.image(image, caption = user_input, use_container_width=True)

    # Afficher les résultats pertinents sur le côté de la page
    st.sidebar.write("User file number :",user_input)
    st.sidebar.write(" ")
    if pourcentage_cancer < 0.55:
          st.sidebar.write("Cancer results :")
          st.sidebar.write("test result : Cancer possibility")
          st.sidebar.write("percentage of accuracy : ", round((1 - pourcentage_cancer),2))
          st.sidebar.write(" ")
     
    if pourcentage_dementia < 0.55:
         st.sidebar.write("Dementia results :")
         st.sidebar.write("test result : Dementia possibility")
         st.sidebar.write("percentage of accuracy : ", round((1 - pourcentage_dementia),2))
         st.sidebar.write(" ")
     
    if pourcentage_parkinson < 0.55:
          st.sidebar.write("Parkinson results :")
          st.sidebar.write("test result : Parkinson possibility")
          st.sidebar.write("percentage of accuracy : ", round((1 - pourcentage_parkinson),2))
          st.sidebar.write(" ")

    if pourcentage_dementia >= 0.55 and pourcentage_cancer >= 0.55 and pourcentage_parkinson >= 0.55:
         st.sidebar.write("No abnormal results dectected")
         st.sidebar.write(" ")


    st.sidebar.write(" ")
    
    if st.sidebar.button("Save to patient's file"):
          user_input = st.write(" ")
           # Créer une instance du patient
          patient = Patient(user_input, resultat_cancer, pourcentage_cancer, resultat_dementia, pourcentage_dementia,
                       resultat_parkinson, pourcentage_parkinson)

          # Créer un fichier csv
          csv_filename = "patients.csv"

          # Enregistre les données du patients dans un fichier csv
          with open(csv_filename, mode="w", newline="") as file:
               field_names = [
                    "File number",
                    "Cancer",
                    "Cancer percentage",
                    "Dementia",
                    "Dementia percentage",
                    "Parkinson",
                    "Parkinson percentage",
               ]
               writer = csv.DictWriter(file, fieldnames=field_names)

               writer.writeheader()
               writer.writerow(patient.to_dict())
   


    
   







    
    


