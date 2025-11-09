"""
Quick test to verify the web app setup
"""

import os
import sys

def test_setup():
    print("🧪 Testing Web App Setup")
    print("=" * 40)
    
    # Check Flask
    try:
        import flask
        print("✅ Flask installed:", flask.__version__)
    except ImportError:
        print("❌ Flask not installed")
        return False
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print("✅ TensorFlow installed:", tf.__version__)
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    
    # Check PIL
    try:
        from PIL import Image
        print("✅ Pillow installed")
    except ImportError:
        print("❌ Pillow not installed")
        return False
    
    # Check numpy
    try:
        import numpy as np
        print("✅ NumPy installed:", np.__version__)
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    # Check model file
    model_path = '../deployment/tomato_disease_model.pkl'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {size_mb:.1f} MB")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Make sure deployment folder is in parent directory")
        return False
    
    # Check templates
    if os.path.exists('templates/index.html'):
        print("✅ Templates folder found")
    else:
        print("❌ Templates folder missing")
        return False
    
    print("=" * 40)
    print("🎉 All checks passed!")
    print("\n🚀 Ready to start the web app!")
    print("   Run: python app.py")
    print("   Then open: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
