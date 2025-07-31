
# Sistema de Clasificación Automática de Residuos Mediante Redes Neuronales Convolucionales

## Descripción General
Este proyecto tiene como objetivo desarrollar un sistema inteligente capaz de clasificar automáticamente diferentes tipos de residuos sólidos urbanos utilizando técnicas de visión por computadora y aprendizaje profundo. El sistema emplea una red neuronal convolucional (ResNet-50) entrenada sobre un dataset público de imágenes de basura, permitiendo identificar siete categorías principales de materiales reciclables.

La solución incluye:
- Un cuaderno Jupyter reproducible para entrenamiento y evaluación del modelo.
- Un script para despliegue en tiempo real usando la cámara web.
- Documentación y archivos de configuración para facilitar la instalación y uso.

## Motivación
La gestión eficiente de residuos es fundamental para la sostenibilidad ambiental. Automatizar la clasificación de basura reduce errores humanos, mejora el reciclaje y contribuye a la reducción de contaminación cruzada entre materiales.

## Estructura del Proyecto

- `Classifier_training.ipynb`: Notebook principal para entrenamiento y evaluación del modelo.
- `realtime_classifier.py`: Script para clasificación en tiempo real usando webcam.
- `propuesta_proyecto_garbage_classification.md`: Documento de propuesta y justificación del proyecto.
- `requirements.txt`: Lista de dependencias necesarias.
- `.gitignore`: Archivos y carpetas excluidos del repositorio.
- `garbage_classification/`: Carpeta con las imágenes del dataset (no incluida en el repositorio por tamaño).
- `mejor_modelo.pth` y `resnet50_Garbage_final.pth`: Pesos del modelo entrenado (no incluidos en el repositorio).

## Requisitos del Sistema y Librerías

- Python 3.8 o superior (recomendado 3.11)
- Sistema operativo: Linux, Windows o MacOS
- GPU opcional para acelerar el entrenamiento

### Principales librerías:
- torch
- torchvision
- opencv-python
- Pillow
- numpy
- scikit-learn
- matplotlib
- tqdm

## Instalación de Dependencias

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Joseph0choa/Garbage.git
   cd Garbage
   ```
2. (Opcional) Crea y activa un entorno conda:
   ```bash
   conda create -n Garbage python=3.11
   conda activate Garbage
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Guía Paso a Paso para Entrenar el Modelo

1. Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) y colócalo en la carpeta `garbage_classification/`.
2. Abre y ejecuta el notebook `Classifier_training.ipynb` para:
   - Preprocesar los datos
   - Definir y entrenar el modelo ResNet-50
   - Validar y evaluar el rendimiento
   - Guardar el modelo entrenado (`mejor_modelo.pth`)

## Cómo Cargar el Modelo Entrenado

En cualquier script, puedes cargar el modelo así:
```python
import torch
from torchvision import models
import torch.nn as nn

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 7)  # 7 clases
model.load_state_dict(torch.load('mejor_modelo.pth', map_location='cpu'))
model.eval()
```

## Ejemplo de Despliegue en Tiempo Real (Webcam)

1. Ejecuta el script:
   ```bash
   python realtime_classifier.py
   ```
2. Se abrirá una ventana mostrando la predicción en tiempo real sobre la imagen capturada por la cámara web. El color del texto corresponde a la categoría detectada.
3. Presiona `ESC` para cerrar la ventana.

## Créditos / Autores

**Cristian Andres Villarreal Orozco**  
**Anderson Joseph Ochoa Trujillo**

Proyecto para la asignatura **Redes Neuronales y Aprendizaje Profundo**
