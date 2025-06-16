import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Keras desde TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

############################### Parámetros
path = "myData"
labelFile = 'labels.csv'
batch_size_val = 50
epochs_val = 30
imageDimesions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2

############################### Carga de imágenes
count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(os.path.join(path, str(count)))
    for y in myPicList:
        curImg = cv2.imread(os.path.join(path, str(count), y))
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")

images = np.array(images)
classNo = np.array(classNo)

############################### División del dataset
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

steps_per_epoch_val = len(X_train) // batch_size_val

print("Data Shapes")
print("Train", X_train.shape, y_train.shape)
print("Validation", X_validation.shape, y_validation.shape)
print("Test", X_test.shape, y_test.shape)

assert (X_train.shape[1:] == imageDimesions), "Las dimensiones de entrenamiento son incorrectas"
assert (X_validation.shape[1:] == imageDimesions), "Las dimensiones de validación son incorrectas"
assert (X_test.shape[1:] == imageDimesions), "Las dimensiones de test son incorrectas"

############################### Leer etiquetas
data = pd.read_csv(labelFile)
print("Data shape:", data.shape)

############################### Visualización de ejemplos
num_of_samples = []
cols = 5
fig, axs = plt.subplots(nrows=noOfClasses, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap='gray')
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(f"{j} - {row['Name']}")
            num_of_samples.append(len(x_selected))

plt.figure(figsize=(12, 4))
plt.bar(range(0, noOfClasses), num_of_samples)
plt.title("Distribución del dataset")
plt.xlabel("Clase")
plt.ylabel("Cantidad")
plt.show()

############################### Preprocesamiento
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

cv2.imshow("GrayScale Sample", X_train[random.randint(0, len(X_train) - 1)])
cv2.waitKey(500)
cv2.destroyAllWindows()

X_train = X_train.reshape(X_train.shape[0], imageDimesions[0], imageDimesions[1], 1)
X_validation = X_validation.reshape(X_validation.shape[0], imageDimesions[0], imageDimesions[1], 1)
X_test = X_test.reshape(X_test.shape[0], imageDimesions[0], imageDimesions[1], 1)

############################### Aumento de datos
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

############################### Modelo
def myModel():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(imageDimesions[0], imageDimesions[1], 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(Conv2D(30, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

############################### Entrenamiento
model = myModel()
print(model.summary())

history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val),
                    steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val,
                    validation_data=(X_validation, y_validation),
                    shuffle=True)

############################### Gráficos de resultado
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.title('Pérdida')
plt.xlabel('Época')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train', 'validation'])
plt.title('Precisión')
plt.xlabel('Época')
plt.show()

############################### Evaluación
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

############################### Guardar modelo
model.save("modelo_trained.h5")
print("Modelo guardado correctamente como modelo_trained.h5")
