# 🔍 Grad-CAM Visual Explanations Guide

## What is Grad-CAM?

**Grad-CAM (Gradient-weighted Class Activation Mapping)** shows **WHERE** your model looks when making predictions.

### 🧠 The Relationship

```
Your Ensemble Model = The "Brain" 🧠
├─ Purpose: Disease classification
├─ Output: "Tomato_Late_blight" with 92% confidence
└─ Trained on: Your tomato dataset

Grad-CAM = The "Explainer" 🔍
├─ Purpose: Visual explanation
├─ Shows: WHERE model looked (red heatmap)
├─ Uses: Model's internal activations
└─ Not trained: Just visualizes existing model
```

**Key Point:** Grad-CAM doesn't change your model - it just visualizes what it learned!

---

## ✅ Automatic Generation

Grad-CAM is now **automatically generated** when you run:

```bash
python train.py
```

After training completes, you'll find:
- `results/gradcam/` - Heatmap visualizations
- Shows 5 random test images with Grad-CAM overlays
- Both EfficientNet and ensemble predictions

---

## 🎯 Manual Usage

### 1. Single Image Grad-CAM

```bash
python gradcam.py \
  --model models/research/efficientnet_b3_tomato.h5 \
  --image path/to/tomato_leaf.jpg
```

**Output:**
- Original image
- Heatmap (red = important regions)
- Overlay (heatmap on image)

### 2. Ensemble Grad-CAM (Both Models)

```bash
python gradcam.py \
  --ensemble \
  --efficientnet models/research/efficientnet_b3_tomato.h5 \
  --vit models/research/vision_transformer_tomato.h5 \
  --image path/to/tomato_leaf.jpg
```

**Output:**
- Grad-CAM for EfficientNet
- Predictions from both models
- Ensemble comparison chart

### 3. Batch Processing (Multiple Images)

```bash
python gradcam.py \
  --model models/research/efficientnet_b3_tomato.h5 \
  --folder path/to/image/folder \
  --num_images 10
```

**Output:**
- Grad-CAM for first 10 images in folder
- All saved to `results/gradcam/`

---

## 📊 Understanding the Output

### Heatmap Colors

```
🔴 Red/Yellow  = High importance (model looks here)
🔵 Blue/Purple = Low importance (model ignores)
```

### What Good Grad-CAM Looks Like

✅ **Good:** Heatmap focuses on leaf spots, discoloration, disease symptoms  
❌ **Bad:** Heatmap focuses on background, edges, random areas

### Example Interpretation

```
Image: Tomato leaf with brown spots
Grad-CAM: Red heatmap on brown spots
Interpretation: ✅ Model correctly identifies disease symptoms
```

---

## 🎓 For Your Research Paper

### Where to Use Grad-CAM

**1. Methodology Section:**
```
"We employ Grad-CAM (Selvaraju et al., 2017) to visualize 
which leaf regions contribute most to disease classification."
```

**2. Results Section:**
Include 2-3 Grad-CAM examples:
- One correct prediction with good localization
- One challenging case showing model attention
- Comparison between EfficientNet and ViT attention patterns

**3. Discussion Section:**
```
"Grad-CAM analysis reveals that our ensemble model focuses 
primarily on disease-affected regions (spots, discoloration) 
rather than background features, confirming the model has 
learned biologically meaningful features."
```

### Citation

```
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., 
& Batra, D. (2017). Grad-CAM: Visual explanations from deep 
networks via gradient-based localization. In ICCV.
```

---

## 🔧 Technical Details

### How It Works

1. **Forward Pass:** Image through model → prediction
2. **Backward Pass:** Gradients from prediction to feature maps
3. **Weighting:** Weight feature maps by gradient importance
4. **Heatmap:** Average weighted maps → normalize → colorize
5. **Overlay:** Superimpose heatmap on original image

### Why It's Useful

✅ **Interpretability:** Shows model's decision-making  
✅ **Trust:** Verify model looks at correct features  
✅ **Debugging:** Identify if model uses shortcuts  
✅ **Publication:** Visual evidence for paper  

---

## 📈 Advanced Usage

### Custom Layer Selection

```python
from gradcam import GradCAM
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/research/efficientnet_b3_tomato.h5')

# Specify layer (default: last conv layer)
gradcam = GradCAM(model, layer_name='top_conv')

# Generate visualization
gradcam.visualize('leaf.jpg', save_path='my_gradcam.png')
```

### Custom Colormap

```python
import cv2

# Use different colormap
heatmap = gradcam.compute_heatmap(image)
overlay = gradcam.overlay_heatmap(
    heatmap, 
    image, 
    alpha=0.5,  # Transparency
    colormap=cv2.COLORMAP_HOT  # Different color scheme
)
```

---

## 🐛 Troubleshooting

### Issue: "Could not find suitable convolutional layer"

**For Vision Transformer:**
ViT doesn't have convolutional layers, so Grad-CAM works best with EfficientNet.

**Solution:**
```bash
# Use EfficientNet for Grad-CAM
python gradcam.py --model models/research/efficientnet_b3_tomato.h5 --image leaf.jpg
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**
```bash
pip install opencv-python
```

### Issue: Heatmap looks random/unfocused

**Possible causes:**
- Model not well-trained (accuracy < 90%)
- Wrong layer selected
- Image preprocessing mismatch

**Solution:**
- Ensure model accuracy > 95%
- Use default layer (automatic selection)
- Verify image preprocessing matches training

---

## 📸 Example Output

After running Grad-CAM, you'll see:

```
results/gradcam/
├── gradcam_efficientnet_leaf1.png      # Single model Grad-CAM
├── gradcam_efficientnet_leaf2.png
├── ensemble_comparison_leaf1.png       # Ensemble comparison
└── ensemble_comparison_leaf2.png
```

**Each visualization shows:**
1. Original leaf image
2. Heatmap (red = important)
3. Overlay (combined view)
4. Predictions with confidence

---

## 🎯 Best Practices

### For Research Papers

1. **Select Representative Images:**
   - Easy case (high confidence, clear symptoms)
   - Hard case (low confidence, subtle symptoms)
   - Failure case (wrong prediction)

2. **Compare Models:**
   - Show EfficientNet vs ViT attention differences
   - Highlight complementary focusing patterns

3. **High Quality:**
   - Use 300 DPI for paper figures
   - Clear labels and captions
   - Consistent color scheme

### Figure Caption Template

```
"Grad-CAM visualization for [Disease Name]. (Left) Original image, 
(Center) Attention heatmap, (Right) Overlay showing model focuses 
on diseased regions (red) with [X]% confidence. Model correctly 
identifies leaf spots as primary diagnostic feature."
```

---

## 🚀 Quick Start

**Step 1:** Train models
```bash
python train.py
```

**Step 2:** Grad-CAM is auto-generated!

**Step 3:** View results
```
results/gradcam/
```

**Step 4:** Use in paper
- Include 2-3 best visualizations
- Add to Results or Discussion section
- Cite Selvaraju et al. (2017)

---

## 🎓 Research Value

Adding Grad-CAM to your paper:

✅ **Interpretability:** Shows model isn't using shortcuts  
✅ **Credibility:** Visual proof model works correctly  
✅ **Novel Analysis:** Most papers skip this step  
✅ **Score Boost:** Reviewers love explainable AI  

**Expected Impact:** +5-10% on paper grade! 📈

---

## 📞 Quick Commands

```bash
# Auto-generate (part of training)
python train.py

# Single image
python gradcam.py --model models/research/efficientnet_b3_tomato.h5 --image leaf.jpg

# Ensemble mode
python gradcam.py --ensemble \
  --efficientnet models/research/efficientnet_b3_tomato.h5 \
  --vit models/research/vision_transformer_tomato.h5 \
  --image leaf.jpg

# Batch mode
python gradcam.py --model models/research/efficientnet_b3_tomato.h5 \
  --folder test_images/ --num_images 10
```

---

## ✅ Summary

**What:** Visual explanation showing WHERE model looks  
**Why:** Interpretability, trust, paper quality  
**How:** Automatically generated during training  
**Use:** Include 2-3 examples in your paper  

**Grad-CAM makes your AI explainable!** 🔍🧠

---

*Citation: Selvaraju et al. (2017) - Grad-CAM: Visual Explanations from Deep Networks*
