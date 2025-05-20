import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Cargar modelo
model_path = os.path.join("..", "model", "best_dermaAI_model.keras")
model = tf.keras.models.load_model(model_path)

# Configuración de rutas y etiquetas
base_test_dir = os.path.join("..", "data", "test")

class_dirs = {'benign': 0, 'malignant': 1}
class_labels = {0: 'no_melanoma', 1: 'melanoma'}

# Métricas
y_true = []
y_pred = []

# Recorrer cada subcarpeta (benign, malignant)
for class_name, label in class_dirs.items():
    class_folder = os.path.join(base_test_dir, class_name)
    for fname in os.listdir(class_folder):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(class_folder, fname)
            
            # Cargar y procesar imagen
            img = image.load_img(img_path, target_size=(300, 300))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predicción
            prediction = model.predict(img_array)
            predicted_class = int(prediction[0][0] > 0.5)

            y_true.append(label)
            y_pred.append(predicted_class)

            print(f"{fname}: Predicción: {class_labels[predicted_class]}, Realidad: {class_labels[label]}")

# Calcular accuracy
accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"\nAccuracy: {accuracy:.2%}")