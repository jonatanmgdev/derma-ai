# DermaAI - Clasificación de Imágenes Dermatológicas

Este proyecto implementa un modelo de Deep Learning para la **detección binaria de melanoma** a partir de imágenes dermatológicas, utilizando arquitecturas preentrenadas como `ResNet50V2` o `EfficientNetB3`.

## 📁 Estructura del Proyecto

```plaintext
dermaAI/
├── data/
│   ├── train/
│   │   ├── benign/
│   │   └── malignant/
│   └── test/
│       ├── benign/
│       └── malignant/
│
├── model/
│   └── best_dermaAI_model.keras
│
├── train.py          # Entrenamiento del modelo
├── predict.py        # Predicciones sobre el set de prueba
└── README.md         # Este archivo
```

## 🚀 Requisitos

Instala los requerimientos necesarios ejecutando:

```bash
pip install -r requirements.txt
```

## 🧠 Entrenamiento
Para entrenar el modelo desde cero, ejecuta:
```bash
python train.py
```
Esto hará lo siguiente:
- Cargará las imágenes desde data/train.
- Aplicará data augmentation.
- Usará ResNet50V2 (o EfficientNetB3 si se cambia en el script).
- Entrenará el modelo con early stopping y reducción de tasa de aprendizaje.
- Guardará el mejor modelo en model/best_dermaAI_model.keras.
- Realizará fine-tuning de las últimas capas del modelo base.


##  📊 Evaluación
Durante el entrenamiento (train.py), se imprime:
- Matriz de Confusión
- Reporte de Clasificación (Precision, Recall, F1-Score) para:
no_melanoma
melanoma

##  🔍 Predicción
```bash
python predict.py
```
Esto realizará inferencias sobre las imágenes en data/test, mostrándote por consola la clase predicha y la real, y calculará la precisión global (accuracy).

### Ejemplo de salida (predicción)
```plaintext
img001.jpg: Predicción: melanoma, Realidad: melanoma
img002.jpg: Predicción: no_melanoma, Realidad: melanoma
...
Accuracy: 87.50%
```

## 🧑‍💻 Autor
Jonatan Montesdeoca González