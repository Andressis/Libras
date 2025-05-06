//atualizar pagina

from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import os
import base64
from flask_cors import CORS
import logging
import traceback  # Adicionado para melhorar logs de erro

# Configurar logging mais detalhado
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Inicialização do modelo e MediaPipe
try:
    # Corrigir o caminho do modelo para usar um caminho absoluto
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keras_model.h5')
    logger.info(f"Tentando carregar modelo de: {model_path}")
    
    # Verificar se o arquivo existe antes de carregar
    if os.path.exists(model_path):
        model = load_model(model_path)
        logger.info("Modelo carregado com sucesso")
    else:
        logger.error(f"Arquivo do modelo não encontrado em: {model_path}")
        # Listar arquivos no diretório para debugging
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Arquivos no diretório: {os.listdir(current_dir)}")
        model = None
except Exception as e:
    logger.error(f"Erro ao carregar o modelo: {str(e)}")
    logger.error(traceback.format_exc())  # Adicionado para capturar a stack trace completa
    model = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W']

# Variável global para armazenar a letra reconhecida mais recente
current_letter = ""

# Rota para verificar o status do servidor e do modelo
@app.route('/status')
def status():
    try:
        # Verificação mais detalhada do modelo
        model_check = model is not None
        model_info = {}
        if model_check:
            model_info = {
                "summary": str(model.summary),
                "config": str(model.get_config())[:100] + "..." # Apenas parte da configuração para não sobrecarregar
            }
            
        return jsonify({
            'status': 'ok',
            'model_loaded': model_check,
            'model_info': model_info,
            'classes': classes,
            'working_dir': os.getcwd(),
            'config_path': model_path if 'model_path' in locals() else "Não definido"
        })
    except Exception as e:
        logger.error(f"Erro na rota de status: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Rota para o index.html
@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Erro ao renderizar index.html: {str(e)}")
        # Se o template não for encontrado, retorne o HTML diretamente
        with open('index.html', 'r') as file:
            return file.read()

# Rota para processar frames recebidos do frontend e reconhecer letras
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global current_letter
    
    try:
        # Receber dados da imagem do frontend
        data = request.json
        if not data or 'image' not in data:
            logger.warning("Dados da imagem não recebidos corretamente")
            return jsonify({'error': 'Dados de imagem não encontrados'}), 400
            
        # Separar o cabeçalho do base64 se existir
        image_data = data['image']
        if ',' in image_data:
            encoded_data = image_data.split(',')[1]
        else:
            encoded_data = image_data
            
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Falha ao decodificar imagem")
            return jsonify({'error': 'Falha ao decodificar imagem'}), 400
        
        # Processamento da imagem com MediaPipe e modelo
        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frameRGB.flags.writeable = False
        results = hands.process(frameRGB)
        frameRGB.flags.writeable = True
        
        # Verificar se há mãos na imagem
        processed_image = img.copy()
        hand_detected = False
        
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                x_max, y_max, x_min, y_min = 0, 0, img.shape[1], img.shape[0]
                for lm in hand.landmark:
                    x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
                    x_max, y_max = max(x_max, x), max(y_max, y)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                
                x_min, y_min = max(0, x_min-50), max(0, y_min-50)
                x_max, y_max = min(img.shape[1], x_max+50), min(img.shape[0], y_max+50)
                
                # Desenhar retângulo verde ao redor da mão
                cv2.rectangle(processed_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                hand_detected = True
                
                # Se o modelo não estiver disponível, não tente predizer
                if model is None:
                    current_letter = "MODELO NÃO CARREGADO"
                    logger.warning("Tentativa de predição sem modelo carregado")
                    cv2.putText(processed_image, "MODELO NÃO CARREGADO", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    imgCrop = img[y_min:y_max, x_min:x_max]
                    if imgCrop.size != 0:
                        imgCrop = cv2.resize(imgCrop, (224, 224))
                        imgArray = np.asarray(imgCrop)
                        normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                        data[0] = normalized_image_array
                        
                        try:
                            prediction = model.predict(data, verbose=0)
                            indexVal = np.argmax(prediction)
                            
                            # Atualizar a letra reconhecida
                            current_letter = classes[indexVal]
                            
                            # Adicionar letra vermelha acima do retângulo verde
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            text_x = x_min + int((x_max - x_min) / 2) - 20
                            text_y = y_min - 30
                            cv2.putText(processed_image, current_letter, (text_x, text_y), 
                                       font, 3, (0, 0, 255), 5, cv2.LINE_AA)
                        except Exception as e:
                            logger.error(f"Erro na predição: {str(e)}")
                            current_letter = "ERRO"
        
        # Converter a imagem processada para base64 para enviar de volta ao front-end
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'letter': current_letter, 
            'processed_image': f'data:image/jpeg;base64,{processed_img_base64}', 
            'hand_detected': hand_detected
        })
    
    except Exception as e:
        logger.error(f"Erro no processamento do frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

# Rota para obter a letra reconhecida atual
@app.route('/get_letter')
def get_letter():
    global current_letter
    return jsonify({'letter': current_letter})

if __name__ == "__main__":
    # Use PORT de variáveis de ambiente ou 5000 por padrão
    port = int(os.environ.get('PORT', 5000))
    # Certifique-se de usar host="0.0.0.0" para aceitar conexões externas
    app.run(host="0.0.0.0", port=port, debug=False)
