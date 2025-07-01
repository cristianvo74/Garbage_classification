# -*- coding: utf-8 -*-
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['brown-glass', 'cardboard', 'green-glass', 'metal', 'paper', 'plastic', 'white-glass']

# Papeleras con colores asociados
papeleras = {
    'brown-glass': ('Canasta Verde', (0, 255, 0)),  # Verde
    'cardboard': ('Canasta Azul', (255, 0, 0)),     # Azul
    'green-glass': ('Canasta Verde', (0, 255, 0)),  # Verde
    'metal': ('Canasta Roja', (0, 0, 255)),         # Rojo
    'paper': ('Canasta Azul', (255, 0, 0)),         # Azul
    'plastic': ('Canasta Amarilla', (0, 255, 255)), # Amarillo
    'white-glass': ('Canasta Verde', (0, 255, 0)),  # Verde
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('resnet50_webcam.pth', map_location=device))
model.to(device)
model.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Crear una ventana normal que pueda ser minimizada
cv2.namedWindow('Clasificador de Residuos', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    input_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label, color = papeleras[class_names[predicted.item()]]

    # Mostrar el texto con el color correspondiente
    cv2.putText(frame, f'Prediccion: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 2)
    cv2.imshow('Clasificador de Residuos', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows() 