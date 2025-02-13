import zipfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import kaggle
DATASET = "dansbecker/food-101"
KAGGLE_JSON_PATH = os.path.expanduser("~/.kaggle/kaggle.json")
if not os.path.exists(KAGGLE_JSON_PATH):
print("Place your kaggle.json file in ~/.kaggle/ directory.")
else:
os.system(f"kaggle datasets download -d {DATASET}")
with zipfile.ZipFile("food-101.zip", 'r') as zip_ref:
zip_ref.extractall("food-101")
print("Dataset extracted successfully!")
data_dir = "food-101/images"
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
data_dir,
target_size=img_size,
batch_size=batch_size,
class_mode='categorical',
subset='training'
)
val_generator = datagen.flow_from_directory(
data_dir,
target_size=img_size,
 batch_size=batch_size,
class_mode='categorical',
subset='validation'
)
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=val_generator, epochs=10)

model.save("food_recognition_model.h5")
print("Model training completed and saved.")
