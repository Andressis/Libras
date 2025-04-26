from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

# Inicialização do Flask
app = Flask(__name__)

# Inicialização do modelo e MediaPipe
model = load_model('keras_model.h5')
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W']

# Captura de vídeo
cap = cv2.VideoCapture(0)

# Variável global para armazenar a letra reconhecida mais recente
current_letter = ""

# Função para gerar frames de vídeo em tempo real
def gen_frames():
    global current_letter
    while True:
        success, img = cap.read()
        if not success:
            break

        # Convertendo para RGB
        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frameRGB.flags.writeable = False
        results = hands.process(frameRGB)
        frameRGB.flags.writeable = True
        
        # Desenhando os pontos das mãos e predizendo a letra
        handsPoints = results.multi_hand_landmarks
        if handsPoints:
            for hand in handsPoints:
                x_max, y_max, x_min, y_min = 0, 0, img.shape[1], img.shape[0]
                for lm in hand.landmark:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    x_max, y_max = max(x_max, x), max(y_max, y)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                
                x_min, y_min = max(0, x_min-50), max(0, y_min-50)
                x_max, y_max = min(img.shape[1], x_max+50), min(img.shape[0], y_max+50)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                imgCrop = img[y_min:y_max, x_min:x_max]
                if imgCrop.size != 0:
                    imgCrop = cv2.resize(imgCrop, (224, 224))
                    imgArray = np.asarray(imgCrop)
                    normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                    data[0] = normalized_image_array
                    prediction = model.predict(data, verbose=0)
                    indexVal = np.argmax(prediction)
                    
                    # Armazenando a letra reconhecida na variável global
                    current_letter = classes[indexVal]
                    cv2.putText(img, current_letter, (x_min, y_min-15), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)

        # Convertendo o frame para o formato adequado para a exibição no frontend
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Rota para o index.html
@app.route('/')
def index():
    return render_template('index.html')

# Rota para gerar o vídeo em tempo real
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Nova rota para obter a letra reconhecida atual
@app.route('/get_letter')
def get_letter():
    global current_letter
    return jsonify({'letter': current_letter})

if __name__ == '__main__':
    app.run(debug=True)