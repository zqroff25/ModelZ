from flask import Flask, jsonify, render_template, request, redirect, url_for
from FireRecognizer import detect_fire_from_image
from SkinCancerRecognizer import detect_skin_cancer
import os
import subprocess

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
@app.route('/run_skin_model', methods=['POST'])
def run_skin_model():
    file = request.files['image']
    filename = 'input.jpg'
    input_path = os.path.join('static/uploads', filename)
    file.save(input_path)

    output_path, label = detect_skin_cancer(input_path)

    return jsonify({
        "input": "uploads/input.jpg",
        "output": "outputs/output.jpg",
        "label": label
    })
from PlateRecognizer import detect_plate_from_image

@app.route('/run_plate_model', methods=['POST'])
def run_plate_model():
    file = request.files['image']
    filename = 'input.jpg'
    input_path = os.path.join('static/uploads', filename)
    file.save(input_path)

    output_path, result_text = detect_plate_from_image(input_path)

    return jsonify({
        "input": "uploads/input.jpg",
        "output": "outputs/output_input.jpg",
        "text": result_text
    })
@app.route('/ask_bot', methods=['POST'])
@app.route('/ask_bot', methods=['POST'])
def ask_bot():
    user_input = request.json.get('message', '')

    system_prompt = """
Senin adın "Z" ve bir yapay zeka asistanısın.
Senin görevlerin:
Sen ModelZ platformunda çalışan bir yapay zeka asistanısın.
Kullanıcılara sadece ModelZ'deki yapay zeka modelleri hakkında bilgi ver.
Konuşma tarzın kısa, net ve teknik bilgilendirici olmalı. Samimi ama cümleler kısa ve doğrudan. Maksimum 2 cümle ile cevap ver.
Kullanıcıya yardımcı olabilmek için sorularını anlamaya çalış. Eğer kullanıcı bir model hakkında bilgi isterse, o modelin ne yaptığını ve nasıl çalıştığını çok kısa maksimum 2 cümle ile açıkla.
Eğer kullanıcı başka konular sorarsa, nazikçe bu konuda yardımcı olamayacağını belirt.
Kullanıcıya nasıl yardımcı olabilirim gibi şık ve ilgili sorular sor. bunlar çok kısa ve net olmalı. Eğer kullanıcı isterse model ile bir demo yapabilmeli.

Şu an ModelZ platformunda bulunan modeller:
- Yangın Tespiti (Haar Cascade)
- Cilt Kanseri Tespiti (CNN)
- Plaka Tanıma ve OCR
- Evo AI Sağlık Asistanı (LLM)
Model dışı sorulara cevap verme. Bu modellerin dışında başka bir model yok. 

Kullanıcının sorusu:
"""


    full_input = system_prompt + user_input

    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1'],
            input=full_input,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )

        response_text = result.stdout.strip()
        return jsonify({'response': response_text})

    except Exception as e:
        return jsonify({'response': f'Hata oluştu: {str(e)}'})


'''if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)'''
if __name__ == '__main__':
    app.run(debug=True, port=8501)
