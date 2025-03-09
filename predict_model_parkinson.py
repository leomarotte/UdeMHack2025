import tensorflow as tf
import numpy as np
from keras.src.legacy.preprocessing import image


IMAGE_SIZE = (224, 224)

# Charger le modèle
model = tf.keras.models.load_model("modele_parkinson.keras")
print("Modèle chargé depuis 'modele_parkinson.keras'.")

def predict_single_image_parkinson(model, img_path):
    # Charger l'image, redimensionnée à la taille du réseau
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    # img = image.load_img(img_path, target_size=IMAGE_SIZE)
    # Convertir en tenseur numpy

    # img_array = image.img_to_array(img)
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Normaliser
    img_array = img_array / 255.0
    # Ajouter la dimension batch
    img_array = np.expand_dims(img_array, axis=0)

    # Effectuer la prédiction
    prediction = model.predict(img_array)[0][0]
    # Seuil = 0.5
    classe = "Parkinson" if prediction <= 0.5 else "No parkinson"
    return classe, prediction

# Test sur une image de votre choix
# Mettez ici le chemin vers l'image
#image_path = "tumor5.jpg"
#classe_predite, score = predict_single_image(model, image_path)
#print(f"Image : {image_path}")
#print(f"Classe prédite : {classe_predite} (score = {score:.4f})")
