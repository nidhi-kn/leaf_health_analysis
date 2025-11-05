"""
Configuration File for Tomato Disease Classification Research
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data' / 'tomato_health'
MODEL_DIR = BASE_DIR / 'models' / 'research'
RESULTS_DIR = BASE_DIR / 'results'
CHECKPOINT_DIR = MODEL_DIR / 'checkpoints'

# Create directories
for dir_path in [MODEL_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Adjust based on your GPU memory
NUM_WORKERS = 4

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 30,
    'width_shift_range': 0.3,
    'height_shift_range': 0.3,
    'shear_range': 0.2,
    'zoom_range': 0.3,
    'horizontal_flip': True,
    'brightness_range': [0.8, 1.2],
    'fill_mode': 'nearest'
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# EfficientNet-B3 Configuration
EFFICIENTNET_CONFIG = {
    'epochs_frozen': 20,        # Increase from 5 to 20 for proper training
    'epochs_finetune': 15,      # Increase from 5 to 15 for proper training
    'freeze_ratio': 0.8,        # Freeze bottom 80% of layers in fine-tuning
    'learning_rate_frozen': 1e-3,
    'learning_rate_finetune': 1e-4,
    'dropout_rate': 0.4,
    'dense_units': 256
}

# Advanced Training Configuration
ADVANCED_CONFIG = {
    'stage3_epochs': 20,        # Gradual unfreezing stage
    'stage4_epochs': 15,        # Full model training stage
    'use_class_weights': True,  # Handle imbalanced dataset
    'use_tta': True,           # Test-time augmentation
    'tta_steps': 5,            # Number of TTA augmentations
    'enhanced_architecture': True,  # Use larger head
    'cyclical_lr': True,       # Use cyclical learning rates
    'gradual_unfreezing': True # Gradual layer unfreezing
}

# Vision Transformer Configuration
VIT_CONFIG = {
    'patch_size': 16,
    'num_heads': 8,
    'transformer_layers': 6,
    'projection_dim': 256,
    'mlp_ratio': 2,
    'epochs': 30,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'dropout_rate': 0.3
}

# Ensemble Configuration (EfficientNet-B3 vs ViT)
ENSEMBLE_CONFIG = {
    'weights': [0.5, 0.5],  # [EfficientNet-B3 weight, ViT weight]
    'strategy': 'weighted_average'  # Options: 'weighted_average', 'voting'
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

TRAINING_CONFIG = {
    'mixed_precision': True,
    'early_stopping_patience': 8,
    'reduce_lr_patience': 4,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7,
    'verbose': 1
}

# ============================================================================
# RESEARCH PAPER CONFIGURATION
# ============================================================================

PAPER_CONFIG = {
    'title': 'CNN vs Transformer for Tomato Disease Classification: A Comparative Study of EfficientNet-B3 and Vision Transformer',
    'abstract_template': """
    We present a comprehensive comparative evaluation of CNN and Transformer architectures for 
    automated tomato disease classification. This study compares the state-of-the-art CNN 
    approach (EfficientNet-B3) with the emerging Vision Transformer (ViT), analyzing their 
    complementary strengths in agricultural AI applications. Using a dataset of {total_images} 
    tomato leaf images across {num_classes} disease classes, we systematically evaluate both 
    architectures and propose an ensemble approach. Our results demonstrate that Vision 
    Transformer achieves {vit_acc}% accuracy compared to {efficientnet_acc}% for EfficientNet-B3, 
    with the ensemble model reaching {ensemble_acc}% accuracy, highlighting the potential of 
    attention-based mechanisms in plant disease detection while demonstrating the continued 
    relevance of convolutional approaches.
    """,
    'keywords': [
        'Deep Learning',
        'Plant Disease Detection',
        'EfficientNet',
        'Vision Transformer',
        'CNN vs Transformer',
        'Attention Mechanism',
        'Ensemble Learning',
        'Agricultural AI',
        'Computer Vision',
        'Transfer Learning'
    ]
}

# ============================================================================
# DISEASE CLASSES
# ============================================================================

DISEASE_CLASSES = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'color_scheme': {
        'efficientnet': '#3498db',
        'vit': '#e74c3c',
        'ensemble': '#2ecc71'
    },
    'plot_training_history': True,
    'plot_confusion_matrix': True,
    'plot_class_activations': True
}

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def setup_gpu():
    """Configure GPU settings"""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ {len(gpus)} GPU(s) available and configured")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU found. Training will use CPU (slower)")
    
    return len(gpus) > 0

# ============================================================================
# MIXED PRECISION
# ============================================================================

def enable_mixed_precision():
    """Enable mixed precision training for faster computation"""
    import tensorflow as tf
    
    if TRAINING_CONFIG['mixed_precision']:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled (faster training)")
    else:
        print("ℹ️  Mixed precision disabled")
