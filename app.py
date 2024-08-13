import os

import streamlit as st # type: ignore
import numpy as np

from keras.preprocessing.image import load_img, img_to_array # type: ignore
from keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder


class CatImageClassifier:
    def __init__(self, model_path, label_encoder_path, image_size=(128, 128)):
        self.model = load_model(model_path)
        self.image_size = image_size
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)

    def predict_cat(self, image_path):
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array)
        predicted_label = self.label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]

classifier = CatImageClassifier('cat_classifier_model.h5', 'label_encoder_classes.npy')

# Streamlit app
st.title("Cat Name Predictor")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open(f"temp_{uploaded_file.name}", "wb") as buffer:
        buffer.write(uploaded_file.getbuffer())
    
    predicted_cat_name = classifier.predict_cat(f"temp_{uploaded_file.name}")
    st.image(f"temp_{uploaded_file.name}", caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Cat Name: {predicted_cat_name}")
    os.remove(f"temp_{uploaded_file.name}")