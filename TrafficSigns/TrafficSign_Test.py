import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

#############################################
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX
#############################################

# Cargar modelo .h5
model = load_model("modelo_trained.h5")

# Leer labels.csv
label_df = pd.read_csv("labels.csv")
label_dict = dict(zip(label_df["ClassId"], label_df["Name"]))

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Funciones de preprocesamiento
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

while True:
    success, imgOriginal = cap.read()

    if not success:
        continue

    # Procesar imagen
    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Predicción
    predictions = model.predict(img)
    classIndex = int(np.argmax(predictions))
    probabilityValue = np.max(predictions)

    if probabilityValue > threshold:
        className = label_dict.get(classIndex, "Desconocido")
        cv2.putText(imgOriginal, f"CLASS: {classIndex} - {className}", (20, 35),
                    font, 0.75, (0, 0, 255), 2)
        cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 75),
                    font, 0.75, (0, 0, 255), 2)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
