import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# Parámetros
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.70
font = cv2.FONT_HERSHEY_SIMPLEX

# Cargar modelo y etiquetas
model = load_model("modelo_trained.h5")
label_dict = dict(zip(pd.read_csv("labels.csv")["ClassId"], pd.read_csv("labels.csv")["Name"]))

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

while True:
    success, imgOriginal = cap.read()
    if not success:
        continue

    # Preprocesamiento directo
    img = cv2.resize(imgOriginal, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = img.reshape(1, 32, 32, 1).astype("float32")

    # Predicción
    predictions = model.predict(img)
    classIndex = int(np.argmax(predictions))
    probabilityValue = np.max(predictions)

    if probabilityValue > threshold:
        className = label_dict.get(classIndex, "Desconocido")
        cv2.putText(imgOriginal, f"CLASS: {classIndex} - {className}",
                    (20, 35), font, 0.75, (0, 0, 255), 2)
        cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%",
                    (20, 75), font, 0.75, (255, 0, 0), 2)

    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
