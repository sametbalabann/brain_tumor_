from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Model yükleme
model_path = '../brain_tumor_model.h5'
model = tf.keras.models.load_model(model_path)

def preprocess_image(image_data):
    """Görüntüyü model için hazırla"""
    
    # Base64'ten görüntü çözme
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
    # RGB'ye çevir
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # 64x64'e resize et
    image = image.resize((64, 64))
        
    # Numpy array'e çevir ve normalize et
    image_array = np.array(image) / 255.0
        
    # Batch dimension ekle
    image_array = np.expand_dims(image_array, axis=0)
        
    return image_array

@app.route('/predict', methods=['POST'])
def predict_tumor():
    try:
        # JSON'dan görüntü verisini al
        data = request.get_json()
        image_data = data['image']  # Base64 encoded image
                
        # Görüntüyü işle
        processed_image = preprocess_image(image_data)
                
        # Tahmin yap
        prediction = model.predict(processed_image)[0][0]
                
        # Sonucu yorumla
        if prediction > 0.5:
            result = "Tümör tespit edildi"
            confidence = float(prediction)
        else:
            result = "Tümör tespit edilmedi"
            confidence = float(1 - prediction)
                
        return jsonify({
            'sonuc': result,
            'guven_orani': round(confidence * 100, 2),
            'ham_skor': float(prediction)
        })
        
    except Exception as e:
        return jsonify({'hata': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'durum': 'API çalışıyor!'})

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>Beyin Tümörü Tahmin API</h1>
    <p>Kullanım:</p>
    <ul>
        <li>POST /predict - MR görüntüsü tahmin etmek için</li>
        <li>GET /health - API durumu kontrol etmek için</li>
    </ul>
    '''

# Vercel için gerekli
def handler(event, context):
    return app(event, context)

