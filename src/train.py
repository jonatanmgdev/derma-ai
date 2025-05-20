import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# Ruta al dataset
dataset_path = os.path.join("..", "data", "train")

# Preprocesamiento con data augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.8, 1.2]
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = val_datagen.flow_from_directory(
    dataset_path,
    target_size=(300, 300),
    batch_size=32,
    class_mode='binary',
    
    subset='validation',
    shuffle=False
)

# Modelo base
model_selected = 'resnet'

if model_selected == 'resnet':
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
elif model_selected == 'efficientnet':
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
else:
    raise ValueError("Opción inválida para model_selected. Usa 'resnet' o 'efficientnet'.")

# Congelar capas base
base_model.trainable = False

# Construcción del modelo
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('../model/best_dermaAI_model.keras', monitor='val_accuracy', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Entrenamiento inicial
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Fine-tuning: descongelar parte del modelo base
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamiento (fine-tuning)
fine_tune_history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Evaluación final
Y_pred = model.predict(validation_generator)
y_pred = (Y_pred > 0.5).astype(int).flatten() 
y_true = validation_generator.classes

print("\nMatriz de Confusión:")
print(confusion_matrix(y_true, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_true, y_pred, target_names=['no_melanoma', 'melanoma']))