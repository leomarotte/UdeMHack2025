import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title('Brain disease detection')

user_input = st.text_input("Enter file name")

upload_file = st.file_uploader (
     "Choisissez un fichier",
     type=["png", "jpg", "jpeg"],
     accept_multiple_files = False
)

if upload_file is not None and user_input is not None:
    image = Image.open(upload_file)
    st.image(image, caption = user_input, use_container_width=True)


