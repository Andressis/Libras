from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os
import base64

# Inicialização do Flask
app = Flask(__name__)

# Inicialização do modelo e MediaPipe
model = load_model('keras_model.h5')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W']

# Variável global para armazenar a letra reconhecida mais recente
current_letter = ""

# Rota para o index.html
@app.route('/')
def index():
    return render_template('index.html')

# Rota para processar frames recebidos do frontend e reconhecer letras
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_letter
    
    # Receber dados da imagem do frontend
    data = request.json
    encoded_data = data['image'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Processamento da imagem com MediaPipe e modelo
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frameRGB.flags.writeable = False
    results = hands.process(frameRGB)
    frameRGB.flags.writeable = True
    
    # Verificar se há mãos na imagem
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            x_max, y_max, x_min, y_min = 0, 0, img.shape[1], img.shape[0]
            for lm in hand.landmark:
                x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                x_max, y_max = max(x_max, x), max(y_max, y)
                x_min, y_min = min(x_min, x), min(y_min, y)
            
            x_min, y_min = max(0, x_min-50), max(0, y_min-50)
            x_max, y_max = min(img.shape[1], x_max+50), min(img.shape[0], y_max+50)
            
            imgCrop = img[y_min:y_max, x_min:x_max]
            if imgCrop.size != 0:
                imgCrop = cv2.resize(imgCrop, (224, 224))
                imgArray = np.asarray(imgCrop)
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image_array
                prediction = model.predict(data, verbose=0)
                indexVal = np.argmax(prediction)
                
                # Atualizar a letra reconhecida
                current_letter = classes[indexVal]
                
    return jsonify({'letter': current_letter})

# Rota para obter a letra reconhecida atual
@app.route('/get_letter')
def get_letter():
    global current_letter
    return jsonify({'letter': current_letter})

if __name__ == '__main__':
    # Definindo a porta de acordo com a variável de ambiente ou 5000 por padrão
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
