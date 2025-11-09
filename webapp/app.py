"""
Flask Web Application for Tomato Disease Detection
"""

from flask import Flask, render_template, request, jsonify, send_file
import pickle
import numpy as np
from PIL import Image
import io
import base64
import os
import sys
import tensorflow as tf
import cv2

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deployment'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Model paths
MOBILENET_PATH = '../models/quick_test/mobilenet_working.h5'
VIT_PATH = '../models/research/vision_transformer_final.h5'

# Class names
CLASS_NAMES = [
    'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
    'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 
    'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy'
]

# Global models
mobilenet_model = None
vit_model = None

def load_models():
    """Load all available models"""
    global mobilenet_model, vit_model
    
    try:
        # Load MobileNet
        if os.path.exists(MOBILENET_PATH):
            mobilenet_model = tf.keras.models.load_model(MOBILENET_PATH, compile=False)
            print("✅ MobileNet loaded successfully")
        else:
            print(f"⚠️  MobileNet not found at {MOBILENET_PATH}")
        
        # Load ViT
        if os.path.exists(VIT_PATH):
            try:
                vit_model = tf.keras.models.load_model(VIT_PATH, compile=False)
                print("✅ ViT loaded successfully")
            except Exception as e:
                print(f"⚠️  ViT loading failed: {e}")
        else:
            print(f"⚠️  ViT not found at {VIT_PATH}")
        
        return mobilenet_model is not None
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_gradcam(model, img_array, pred_index):
    """Generate Grad-CAM heatmap"""
    try:
        # Find last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                last_conv_layer = layer
                break
            elif 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None
        
        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [last_conv_layer.output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_channel = predictions[:, pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

@app.route('/')
def index():
    """Home page"""
    return render_template('index_enhanced.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        model_type = request.form.get('model', 'mobilenet')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize for display
        display_image = image.copy()
        display_image.thumbnail((400, 400))
        
        # Convert to base64 for display
        buffered = io.BytesIO()
        display_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare for prediction
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized, dtype=np.float32)
        
        # Apply ImageNet preprocessing (MobileNet was trained with this)
        # Normalize to [-1, 1] range for MobileNet
        image_array = (image_array / 127.5) - 1.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Select model and make prediction
        predictions = None
        
        if model_type == 'mobilenet' and mobilenet_model:
            model = mobilenet_model
            model_name = 'MobileNetV2'
            model_accuracy = 86.12
            predictions = model.predict(image_array, verbose=0)
            
        elif model_type == 'vit' and vit_model:
            model = vit_model
            model_name = 'Vision Transformer'
            model_accuracy = 46.96
            predictions = model.predict(image_array, verbose=0)
            
        elif model_type == 'ensemble' and mobilenet_model and vit_model:
            # Ensemble prediction
            mobilenet_pred = mobilenet_model.predict(image_array, verbose=0)
            vit_pred = vit_model.predict(image_array, verbose=0)
            predictions = 0.7 * mobilenet_pred + 0.3 * vit_pred
            model = mobilenet_model  # Use for Grad-CAM
            model_name = 'Ensemble (MobileNet + ViT)'
            model_accuracy = 85.65
            
        else:
            # Default to MobileNet
            model = mobilenet_model
            model_name = 'MobileNetV2'
            model_accuracy = 86.12
            predictions = model.predict(image_array, verbose=0)
        
        # Ensure predictions is not None
        if predictions is None:
            return jsonify({'error': 'Model prediction failed'}), 500
        
        probabilities = tf.nn.softmax(predictions[0]).numpy()
        predicted_idx = np.argmax(probabilities)
        confidence = float(probabilities[predicted_idx])
        
        # Generate Grad-CAM
        heatmap = generate_gradcam(model, image_array, predicted_idx)
        gradcam_str = None
        heatmap_str = None
        
        if heatmap is not None:
            # Resize heatmap to match image
            heatmap_resized = cv2.resize(heatmap, (224, 224))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            
            # Create overlay
            original_img = np.array(image_resized)
            overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', overlay)
            gradcam_str = base64.b64encode(buffer).decode()
            
            # Also create standalone heatmap
            _, heatmap_buffer = cv2.imencode('.jpg', heatmap_colored)
            heatmap_str = base64.b64encode(heatmap_buffer).decode()
        
        # Top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3 = [
            {
                'class': CLASS_NAMES[idx],
                'confidence': round(float(probabilities[idx]) * 100, 2)
            }
            for idx in top_3_idx
        ]
        
        # Format response
        response = {
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_idx],
            'confidence': round(confidence * 100, 2),
            'top_3_predictions': top_3,
            'image': f"data:image/jpeg;base64,{img_str}",
            'gradcam_overlay': f"data:image/jpeg;base64,{gradcam_str}" if gradcam_str else None,
            'gradcam_heatmap': f"data:image/jpeg;base64,{heatmap_str}" if heatmap is not None else None,
            'model_info': {
                'name': model_name,
                'accuracy': model_accuracy,
                'type': 'CNN' if 'MobileNet' in model_name else 'Transformer' if 'ViT' in model_name else 'Ensemble'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/model-info')
def model_info():
    """Get model information"""
    return jsonify({
        'mobilenet': {
            'available': mobilenet_model is not None,
            'accuracy': 86.12,
            'name': 'MobileNetV2'
        },
        'vit': {
            'available': vit_model is not None,
            'accuracy': 46.96,
            'name': 'Vision Transformer'
        },
        'ensemble': {
            'available': mobilenet_model is not None and vit_model is not None,
            'accuracy': 85.65,
            'name': 'Ensemble'
        },
        'classes': CLASS_NAMES
    })

@app.route('/performance-charts')
def performance_charts():
    """Serve performance visualization images"""
    chart_type = request.args.get('type', 'comparison')
    
    if chart_type == 'comparison':
        path = '../model_comparison.png'
    elif chart_type == 'confusion':
        path = '../confusion_matrix.png'
    elif chart_type == 'gradcam':
        path = '../real_leaf_gradcam.png'
    else:
        return jsonify({'error': 'Invalid chart type'}), 400
    
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    else:
        return jsonify({'error': 'Chart not found'}), 404

if __name__ == '__main__':
    print("🚀 Starting Tomato Disease Detection Web App...")
    if load_models():
        print("🌐 Server starting on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load models. Please check the model paths.")
