import tensorflow as tf
from  tensorflow import keras
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",  
    target_size=IMG_SIZE,  
    batch_size=BATCH_SIZE,
    class_mode="binary",  
    subset="training"
)
val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

model_cancer = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_cancer.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

model_cancer.summary()


# history = model_cancer.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=20
# )