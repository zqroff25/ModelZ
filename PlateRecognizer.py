import cv2
import pytesseract
import os

# Tesseract yolu (gerekirse sistemine göre ayarla)
pytesseract.pytesseract.tesseract_cmd = "C:/Users/w11/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

CASCADE_PATH = "Plaka_Model.xml"
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

def detect_plate_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Görsel yüklenemedi")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    result_text = ""
    for (x, y, w, h) in plates:
        roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')
        result_text = text.strip()

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    filename = os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_FOLDER, f"output_{filename}")
    cv2.imwrite(output_path, img)

    return output_path, result_text
#detect_plate_from_image('ozel-plaka.jpg')