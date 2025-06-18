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

# Preprocesamiento robusto, igual que en test.py
def preprocessing(img_pil):
    img = np.array(img_pil)

    # Convertir de RGB a BGR (porque cv2 espera BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Redimensionar a 32x32
    img = cv2.resize(img, (32, 32))

    # Escala de grises
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ecualizar histograma y normalizar
    img = cv2.equalizeHist(img)
    img = img / 255.0

    # Redimensionar a (1, 32, 32, 1)
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
