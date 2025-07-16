from flask import Flask, render_template, request, jsonify
from flask_sock import Sock
import numpy as np
import joblib
from PIL import Image
import io
import base64
import json
from preprocess import preprocess_image
import os
import traceback

app = Flask(__name__)
sock = Sock(app)

# Load model and model info
try:
    model = joblib.load('mnist_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json.get('image') if request.json else None
        if not data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Preprocess image
        arr = preprocess_image(data)
        
        # Get predictions
        probs = model.predict_proba(arr)[0]
        prediction = model.predict(arr)[0]
        
        # Calculate confidence
        confidence = float(np.max(probs))
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': probs.tolist()
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@sock.route('/ws')
def ws(ws):
    while True:
        try:
            data = ws.receive()
            if not data:
                break
            
            if model is None:
                ws.send(json.dumps({'error': 'Model not loaded'}))
                continue
            
            # Preprocess image using the new function
            arr = preprocess_image(data)
            
            # Check if preprocessing worked
            if arr is None or arr.size == 0:
                ws.send(json.dumps({'error': 'Preprocessing failed'}))
                continue
            
            # Get predictions
            probs = model.predict_proba(arr)[0]
            prediction = model.predict(arr)[0]
            confidence = float(np.max(probs))
            
            # Validate the results
            if np.isnan(confidence) or np.isinf(confidence):
                ws.send(json.dumps({'error': 'Invalid prediction values'}))
                continue
            
            response = {
                'probabilities': probs.tolist(),
                'prediction': int(prediction),
                'confidence': confidence
            }
            
            print(f"Prediction: {prediction}, Confidence: {confidence}")
            ws.send(json.dumps(response))
            
        except Exception as e:
            print(f"WebSocket error: {e}")
            traceback.print_exc()
            ws.send(json.dumps({'error': str(e)}))

if __name__ == '__main__':
    app.run(debug=True, port=5000)