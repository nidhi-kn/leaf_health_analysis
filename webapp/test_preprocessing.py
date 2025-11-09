"""
Test preprocessing to ensure it matches training
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import os

# Add parent to path
sys.path.append('..')

# Load model
model_path = '../models/quick_test/mobilenet_working.h5'
model = tf.keras.models.load_model(model_path, compile=False)

print("Model loaded successfully")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Test with a sample image from dataset
test_image_path = '../dataset/PlantVillage/Tomato_Early_blight'

if os.path.exists(test_image_path):
    # Get first image
    images = [f for f in os.listdir(test_image_path) if f.endswith('.JPG') or f.endswith('.jpg')]
    if images:
        img_path = os.path.join(test_image_path, images[0])
        print(f"\nTesting with: {images[0]}")
        
        # Load and preprocess
        image = Image.open(img_path).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized, dtype=np.float32)
        
        # Try different preprocessing methods
        print("\n1. Testing with [0,1] normalization:")
        img_01 = image_array / 255.0
        img_01 = np.expand_dims(img_01, axis=0)
        pred_01 = model.predict(img_01, verbose=0)
        prob_01 = tf.nn.softmax(pred_01[0]).numpy()
        top_01 = np.argmax(prob_01)
        print(f"   Top prediction index: {top_01}, confidence: {prob_01[top_01]*100:.2f}%")
        
        print("\n2. Testing with [-1,1] normalization (ImageNet):")
        img_11 = (image_array / 127.5) - 1.0
        img_11 = np.expand_dims(img_11, axis=0)
        pred_11 = model.predict(img_11, verbose=0)
        prob_11 = tf.nn.softmax(pred_11[0]).numpy()
        top_11 = np.argmax(prob_11)
        print(f"   Top prediction index: {top_11}, confidence: {prob_11[top_11]*100:.2f}%")
        
        print("\n3. Testing with no normalization:")
        img_raw = np.expand_dims(image_array, axis=0)
        pred_raw = model.predict(img_raw, verbose=0)
        prob_raw = tf.nn.softmax(pred_raw[0]).numpy()
        top_raw = np.argmax(prob_raw)
        print(f"   Top prediction index: {top_raw}, confidence: {prob_raw[top_raw]*100:.2f}%")
        
        # Class names
        class_names = [
            'Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
            'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 
            'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy'
        ]
        
        print("\n" + "="*50)
        print("RESULTS:")
        print(f"Expected: Early Blight (index 1)")
        print(f"[0,1] normalization: {class_names[top_01]} ({prob_01[top_01]*100:.2f}%)")
        print(f"[-1,1] normalization: {class_names[top_11]} ({prob_11[top_11]*100:.2f}%)")
        print(f"No normalization: {class_names[top_raw]} ({prob_raw[top_raw]*100:.2f}%)")
        
        # Show top 3 for best method
        best_prob = prob_11 if prob_11[top_11] > prob_01[top_01] else prob_01
        best_name = "[-1,1]" if prob_11[top_11] > prob_01[top_01] else "[0,1]"
        
        print(f"\nTop 3 predictions using {best_name}:")
        top_3_idx = np.argsort(best_prob)[-3:][::-1]
        for i, idx in enumerate(top_3_idx, 1):
            print(f"  {i}. {class_names[idx]}: {best_prob[idx]*100:.2f}%")
        
else:
    print("Test image path not found")
