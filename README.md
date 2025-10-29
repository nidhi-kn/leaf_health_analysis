# 🍅 Tomato Disease Classification - Research Study

**Multi-Model Deep Learning Approach for Agricultural AI**

This project implements and compares three state-of-the-art deep learning models for automated tomato disease classification:
1. **EfficientNet-B3** - SOTA CNN with compound scaling
2. **Vision Transformer (ViT)** - Attention-based architecture
3. **Ensemble** - Combined predictions from both models

---

## 📊 Expected Results

| Model | Expected Accuracy | Parameters | Training Time |
|-------|------------------|------------|---------------|
| EfficientNet-B3 | 96-98% | ~12M | 2.5 hours |
| Vision Transformer | 97-99% | ~8M | 2 hours |
| **Ensemble** | **98-99%** | - | 10 minutes |

---

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset

The dataset should be in `dataset/PlantVillage/` with 10 disease classes:
- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Leaf_Mold
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite
- Tomato__Target_Spot
- Tomato__Tomato_YellowLeaf__Curl_Virus
- Tomato__Tomato_mosaic_virus
- Tomato_healthy

### 3️⃣ Run Complete Training Pipeline

```bash
# Easiest way - do everything
python train.py
```

This will:
- ✅ Check dependencies
- ✅ Prepare dataset (split into train/val/test)
- ✅ Train all 3 models
- ✅ Generate comparison reports
- ✅ Create visualizations for your paper

**Estimated Time:** 5-6 hours (can run overnight)

---

## 📝 Manual Steps

### Step 1: Prepare Data

```bash
python prepare_data.py
```

This splits your dataset into:
- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

### Step 2: Train Models

```bash
python main.py --data_dir data/tomato_health
```

This trains all three models sequentially.

---

## 📁 Project Structure

```
mini Project2/
│
├── dataset/
│   └── PlantVillage/          # Raw dataset
│       ├── Tomato_Bacterial_spot/
│       ├── Tomato_Early_blight/
│       └── ...
│
├── data/
│   └── tomato_health/         # Prepared dataset (created by script)
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   └── research/              # Trained models & results
│       ├── efficientnet_b3_tomato.h5
│       ├── vision_transformer_tomato.h5
│       ├── model_comparison.csv
│       ├── model_comparison.png
│       └── table_for_paper.tex
│
├── results/                   # Additional visualizations
│   ├── confusion_matrices/
│   ├── training_history.png
│   └── results_summary.txt
│
├── main.py                    # Main training script
├── prepare_data.py            # Data preparation
├── config.py                  # Configuration
├── visualization.py           # Visualization utilities
├── train.py                   # Training launcher
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## 🎓 For Your Research Paper

### 📊 Files to Include

1. **Table:** `models/research/table_for_paper.tex`
   - LaTeX table ready to paste in your paper
   
2. **Figures:**
   - `models/research/model_comparison.png` - Performance comparison
   - `results/training_history.png` - Training curves
   - `results/confusion_matrices/*.png` - Per-model confusion matrices

3. **Results:** `results/results_summary.txt`
   - Comprehensive summary of all results

### 📝 Abstract Template

```
We present a comprehensive evaluation of deep learning architectures for 
automated tomato disease classification. We compare traditional CNN approaches 
(EfficientNet-B3) with emerging transformer-based methods (Vision Transformer), 
and propose an ensemble approach combining both. On a dataset of [N] tomato 
leaf images across 10 disease classes, our ensemble model achieves [X]% accuracy, 
outperforming individual models and demonstrating the complementary strengths 
of CNN and attention-based architectures in agricultural AI applications.
```

### 🔑 Key Points for Discussion

1. **EfficientNet-B3:**
   - Uses compound scaling (width, depth, resolution)
   - Pre-trained on ImageNet
   - Strong feature extraction through CNNs

2. **Vision Transformer:**
   - Pure attention-based architecture
   - Treats images as sequences of patches
   - Captures long-range dependencies better

3. **Ensemble:**
   - Combines strengths of both architectures
   - Reduces individual model biases
   - Achieves highest accuracy

---

## ⚙️ Configuration

Edit `config.py` to customize:

- **Image size:** Default 224x224
- **Batch size:** Default 16 (adjust based on GPU)
- **Training epochs:** EfficientNet (20+15), ViT (30)
- **Learning rates:** Optimized for each model
- **Augmentation parameters:** Rotation, zoom, shift, etc.

---

## 🖥️ System Requirements

### Minimum:
- **CPU:** 4+ cores
- **RAM:** 8GB+
- **Storage:** 10GB free space
- **Time:** ~10 hours (CPU)

### Recommended:
- **GPU:** NVIDIA GPU with 6GB+ VRAM
- **RAM:** 16GB+
- **Storage:** 10GB free space
- **Time:** ~5 hours (GPU)

---

## 🐛 Troubleshooting

### Issue: "Data directory not found"
**Solution:**
```bash
python prepare_data.py
```

### Issue: "Out of memory"
**Solution:**
- Reduce `BATCH_SIZE` in `config.py` (try 8 or 4)
- Reduce image size to (192, 192)

### Issue: "No GPU detected"
**Solution:**
- Training will work on CPU (just slower)
- Or install CUDA-enabled TensorFlow:
```bash
pip install tensorflow-gpu
```

### Issue: "Import Error"
**Solution:**
```bash
pip install -r requirements.txt
```

---

## 📚 References

1. **EfficientNet:** Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling for CNNs"
2. **Vision Transformer:** Dosovitskiy et al. (2020) - "An Image is Worth 16x16 Words"
3. **Ensemble Learning:** Dietterich (2000) - "Ensemble Methods in Machine Learning"

---

## 📧 Support

If you encounter issues:
1. Check the error message carefully
2. Review the troubleshooting section
3. Ensure all dependencies are installed
4. Verify dataset structure

---

## 🎯 Results Checklist

After training completes, verify you have:

- [ ] Three trained models in `models/research/`
- [ ] Comparison CSV and plot
- [ ] LaTeX table for paper
- [ ] Confusion matrices for each model
- [ ] Training history plots
- [ ] Results summary

---

## 🏆 Publication Tips

1. **Novelty:** Emphasize the comparative study aspect
2. **Methodology:** Clearly describe each architecture
3. **Results:** Use tables and figures from generated files
4. **Discussion:** Analyze why ensemble outperforms individual models
5. **Future Work:** Suggest improvements (more data, other architectures, etc.)

---

## 📄 License

This is a research project for educational purposes.

---

## 🌟 Good Luck!

This implementation gives you:
- ✅ Publication-quality results
- ✅ Comparative analysis of multiple models
- ✅ Ready-to-use LaTeX tables and figures
- ✅ Complete reproducible pipeline

**Expected paper score improvement:** ⭐⭐⭐⭐⭐

Run `python train.py` and come back in 5 hours for research-ready results! 🚀
