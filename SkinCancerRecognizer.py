import cv2
import numpy as np
import os
from keras.models import load_model

# Model ve resim boyutu
MODEL_PATH = 'skin_cancer_detection_model.h5'
IMG_SIZE = (224, 224)

# Modeli yükle
model = load_model(MODEL_PATH)

def detect_skin_cancer(image_path):
    label = ''
    color = (0, 255, 0)  # varsayılan yeşil (kansersiz)

    # Görseli oku
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Görsel yüklenemedi")

    # Preprocessing
    img = cv2.resize(frame, IMG_SIZE)
    img_input = np.expand_dims(img, axis=0)
    img_input = img_input / 255.0

    # Tahmin yap
    prediction = model.predict(img_input)
    if prediction[0] > 0.5:
        label = 'Kanserli Cilt'
        color = (0, 0, 255)  # kırmızı
    else:
        label = 'Kansersiz Cilt'

    # Görselin üzerine yazı ve kutu çiz
    h, w, _ = frame.shape
    cv2.rectangle(frame, (30, 30), (w - 30, h - 30), color, 4)
    cv2.putText(frame, label, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Sonucu kaydet
    output_path = os.path.join("static", "outputs", "output.jpg")
    cv2.imwrite(output_path, frame)

    return output_path, label