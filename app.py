from flask import Flask, jsonify, render_template, request, redirect, url_for
from FireRecognizer import detect_fire_from_image
import os

app = Flask(__name__)
UPLOAD_PATH = 'static/uploads/input.jpg'
OUTPUT_PATH = 'static/outputs/output.jpg'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # 👇 Görseli sabit isimle kaydet
    file.save(UPLOAD_PATH)

    # Modeli çalıştır ve sonucu sabit olarak üret
    _, detected = detect_fire_from_image(UPLOAD_PATH)

    return jsonify({
        "input": "uploads/input.jpg",
        "output": "outputs/output.jpg",
        "detected": detected
    })
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
'''if __name__ == '__main__':
    app.run(debug=True, port=8501)'''
