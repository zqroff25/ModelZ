import cv2
import os

CASCADE_PATH = 'cascade.xml'

def detect_fire_from_image(image_path):
    object_name = 'Yangın'
    color = (0, 0, 255)
    min_area = 30000

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Görsel yüklenemedi")

    img = cv2.resize(img, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise IOError("Cascade dosyası yüklenemedi!")

    objects = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)

    detected = False
    for (x, y, w, h) in objects:
        if w * h > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, object_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            detected = True

    # 👇 Çıktı sabit dosya
    cv2.imwrite("static/outputs/output.jpg", img)

    return "static/outputs/output.jpg", detected
