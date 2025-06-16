import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Configuración de la página
st.set_page_config(page_title="Clasificador de Señales de Tránsito", layout="centered")
st.title("🚦 Clasificador de Señales de Tránsito")
st.write("Sube una imagen o usa la cámara para identificar la señal.")

# Cachear modelo y etiquetas
@st.cache_resource
def load_model_and_labels():
    model = load_model("TrafficSigns/modelo_trained.h5")
    labels = pd.read_csv("TrafficSigns/labels.csv")
    label_dict = dict(zip(labels["ClassId"], labels["Name"]))
    return model, label_dict

model, label_dict = load_model_and_labels()

# Preprocesamiento robusto
def preprocessing(img_pil):
    img = img_pil.resize((32, 32))
    img = np.array(img)

    # Convertir a escala de grises si es RGB
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.equalizeHist(img)
    img = img / 255.0
    return img.reshape(1, 32, 32, 1).astype("float32")

# Entrada de imagen
img_input = st.camera_input("Toma una foto de una señal") or st.file_uploader("...o sube una imagen", type=["png", "jpg", "jpeg"])

if img_input is not None:
    image = Image.open(img_input)
    img_np = preprocessing(image)

    # Predicción
    predictions = model.predict(img_np)
    classIndex = int(np.argmax(predictions))
    probabilityValue = float(np.max(predictions))
    className = label_dict.get(classIndex, "Desconocido")

    # Mostrar resultados
    st.image(image, caption="Imagen analizada", use_container_width=True)
    st.subheader("🧠 Predicción")
    st.write(f"**Clase:** {classIndex} — {className}")
    st.write(f"**Probabilidad:** {round(probabilityValue * 100, 2)} %")

    if probabilityValue < 0.75:
        st.warning("⚠️ La probabilidad es baja. Intenta tomar una foto más clara.")
