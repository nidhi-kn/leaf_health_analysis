# Tomato Disease Classification - CNN vs Transformer Study

A comprehensive research project comparing CNN (MobileNetV2) and Vision Transformer architectures for automated tomato disease classification.

## 🎯 Project Overview

This project implements and compares two state-of-the-art deep learning approaches:
- **CNN Architecture**: MobileNetV2 (efficient for mobile deployment)
- **Transformer Architecture**: Vision Transformer (attention-based)
- **Ensemble Method**: Combining both approaches

## 📊 Current Results

- **MobileNetV2**: 86.12% accuracy (5 epochs training)
- **Top-3 Accuracy**: 98.09%
- **Inference Speed**: 40.8 samples/sec
- **Status**: Ready for research publication

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
```bash
python 0_prepare_dataset.py
```

### 3. Train Models
```bash
python 1_train_cnn_vs_transformer.py
```

### 4. Test Model
```bash
python 2_test_model.py
```

### 5. Evaluate Results
```bash
python 3_evaluate_model.py
```

## 📁 Project Structure

```
├── 0_prepare_dataset.py      # Dataset preparation and splitting
├── 1_train_cnn_vs_transformer.py  # Main training script (CNN vs Transformer)
├── 2_test_model.py           # Model testing and validation
├── 3_evaluate_model.py       # Comprehensive evaluation
├── config.py                 # Configuration settings
├── visualization.py          # Plotting and visualization utilities
├── gradcam.py               # Grad-CAM visualization for model interpretability
├── utils_save_model.py      # Model saving utilities
├── utils_load_model.py      # Model loading utilities
├── requirements.txt         # Python dependencies
├── data/                    # Processed datasets
│   ├── tomato_health/       # Main dataset (train/val/test split)
│   └── tomato_health_reduced/  # Reduced dataset for quick experiments
├── dataset/                 # Original PlantVillage dataset
├── models/                  # Trained models
│   ├── quick_test/         # Quick test models
│   └── research/           # Research-quality models
└── trash/                  # Archived old files
```

## 🔬 Research Features

### Models Implemented
1. **MobileNetV2**: Efficient CNN with depthwise separable convolutions
2. **Vision Transformer**: Pure attention mechanism for image classification
3. **Ensemble**: Weighted combination of CNN and Transformer predictions

### Advanced Techniques
- Transfer learning from ImageNet
- Advanced data augmentation
- Class balancing for imbalanced datasets
- Test-time augmentation (TTA)
- Grad-CAM visualization for model interpretability

### Evaluation Metrics
- Accuracy and Top-3/Top-5 accuracy
- Per-class performance analysis
- Confusion matrices
- Confidence analysis
- Inference speed benchmarking

## 📈 Dataset Information

- **Source**: PlantVillage Dataset
- **Classes**: 10 tomato disease categories
- **Total Images**: ~23,000 images
- **Split**: 70% train, 15% validation, 15% test
- **Preprocessing**: Resized to 224x224, normalized

### Disease Classes
1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites (Two-spotted)
7. Target Spot
8. Yellow Leaf Curl Virus
9. Mosaic Virus
10. Healthy

## 🎓 Research Applications

This project is suitable for:
- **Academic Research**: CNN vs Transformer comparative studies
- **Agricultural AI**: Real-world plant disease detection
- **Mobile Deployment**: Efficient models for field use
- **Publication**: Conference/journal paper submission

## 🛠️ Technical Requirements

- Python 3.8+
- TensorFlow 2.x
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ storage space

## 📊 Performance Benchmarks

| Model | Accuracy | Top-3 Acc | Parameters | Speed (samples/sec) |
|-------|----------|------------|------------|-------------------|
| MobileNetV2 | 86.12% | 98.09% | 2.4M | 40.8 |
| Vision Transformer | TBD | TBD | ~8M | TBD |
| Ensemble | TBD | TBD | Combined | TBD |

## 🔧 Configuration

Edit `config.py` to customize:
- Model architectures
- Training parameters
- Data augmentation settings
- Evaluation metrics

## 📝 Usage Examples

### Quick Model Test
```python
from utils_load_model import ModelLoader

loader = ModelLoader()
loader.load_model()
result = loader.predict_single_image('path/to/leaf.jpg')
print(f"Prediction: {result['predicted_name']} ({result['confidence']*100:.1f}%)")
```

### Batch Prediction
```python
results = loader.predict_batch(['image1.jpg', 'image2.jpg'])
for result in results:
    print(f"{result['image_path']}: {result['predicted_name']}")
```

## 🎯 Next Steps

1. **Complete Training**: Run full CNN vs Transformer comparison
2. **Hyperparameter Tuning**: Optimize model performance
3. **Deployment**: Create mobile/web application
4. **Research Paper**: Document findings and methodology

## 📚 References

- MobileNetV2: Sandler et al. (2018)
- Vision Transformer: Dosovitskiy et al. (2020)
- PlantVillage Dataset: Hughes & Salathé (2015)

## 🤝 Contributing

This is a research project. For questions or collaboration:
1. Review the code structure
2. Check configuration settings
3. Run experiments with different parameters
4. Document findings and improvements

---

**Status**: Active Development | **Last Updated**: November 2025