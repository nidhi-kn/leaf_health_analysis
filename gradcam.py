"""
Grad-CAM Implementation for Visual Explanations
Shows WHERE the model looks when making predictions

Citation: Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks"
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM)
    
    Visualizes which parts of an image are important for model's prediction
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            layer_name: Name of layer to visualize (default: last conv layer)
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        
        print(f"✅ Grad-CAM initialized with layer: {self.layer_name}")
    
    def _find_target_layer(self):
        """
        Automatically find the last convolutional layer
        """
        # For EfficientNet and similar models
        for layer in reversed(self.model.layers):
            # Check if layer has 4D output (batch, height, width, channels)
            if len(layer.output_shape) == 4:
                return layer.name
        
        # Fallback: try to find common layer names
        common_names = ['top_conv', 'conv_head', 'block7', 'block6']
        for name in common_names:
            for layer in self.model.layers:
                if name in layer.name.lower():
                    return layer.name
        
        raise ValueError("Could not find suitable convolutional layer for Grad-CAM")
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute Grad-CAM heatmap
        
        Args:
            image: Input image (preprocessed, shape: (1, H, W, 3))
            class_idx: Target class index (default: predicted class)
            eps: Small value to avoid division by zero
            
        Returns:
            heatmap: Grad-CAM heatmap (values 0-1)
        """
        # Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            
            # If no class specified, use predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get score for target class
            class_channel = predictions[:, class_idx]
        
        # Compute gradients of class score w.r.t. feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap = heatmap / (np.max(heatmap) + eps)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap
            image: Original image (PIL Image or numpy array)
            alpha: Transparency of heatmap (0-1)
            colormap: OpenCV colormap
            
        Returns:
            superimposed_img: Image with heatmap overlay
        """
        # Convert image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on image
        superimposed_img = heatmap * alpha + image * (1 - alpha)
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img
    
    def visualize(self, image_path, class_idx=None, save_path=None):
        """
        Complete Grad-CAM visualization pipeline
        
        Args:
            image_path: Path to input image
            class_idx: Target class (default: predicted class)
            save_path: Path to save visualization
            
        Returns:
            Dictionary with results
        """
        # Load and preprocess image
        original_img = Image.open(image_path).convert('RGB')
        original_img = original_img.resize((224, 224))
        
        img_array = np.array(original_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = predicted_class
        
        # Compute heatmap
        heatmap = self.compute_heatmap(img_array, class_idx)
        
        # Create overlay
        superimposed = self.overlay_heatmap(heatmap, np.array(original_img))
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap only
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Superimposed
        axes[2].imshow(superimposed)
        axes[2].set_title(f'Grad-CAM Overlay\nClass: {class_idx} ({confidence*100:.2f}%)', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Grad-CAM visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return {
            'predicted_class': int(predicted_class),
            'confidence': float(confidence),
            'heatmap': heatmap,
            'overlay': superimposed
        }


class EnsembleGradCAM:
    """
    Grad-CAM for Ensemble Models
    
    Shows visualizations from both CNN and Transformer components
    """
    
    def __init__(self, efficientnet_model, vit_model, class_names):
        """
        Initialize Ensemble Grad-CAM
        
        Args:
            efficientnet_model: Trained EfficientNet model
            vit_model: Trained Vision Transformer model
            class_names: List of class names
        """
        self.efficientnet = efficientnet_model
        self.vit = vit_model
        self.class_names = class_names
        
        # Create Grad-CAM for EfficientNet
        try:
            self.gradcam_eff = GradCAM(efficientnet_model)
            print("✅ Grad-CAM initialized for EfficientNet")
        except Exception as e:
            print(f"⚠️  Could not initialize Grad-CAM for EfficientNet: {e}")
            self.gradcam_eff = None
    
    def visualize_ensemble(self, image_path, save_dir='results/gradcam'):
        """
        Visualize Grad-CAM for ensemble components
        
        Args:
            image_path: Path to input image
            save_dir: Directory to save visualizations
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and preprocess image
        original_img = Image.open(image_path).convert('RGB')
        original_img_resized = original_img.resize((224, 224))
        
        img_array = np.array(original_img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions from both models
        pred_eff = self.efficientnet.predict(img_array, verbose=0)
        pred_vit = self.vit.predict(img_array, verbose=0)
        pred_ensemble = (pred_eff + pred_vit) / 2.0
        
        # Get predicted classes
        class_eff = np.argmax(pred_eff[0])
        class_vit = np.argmax(pred_vit[0])
        class_ensemble = np.argmax(pred_ensemble[0])
        
        print("\n" + "="*70)
        print("ENSEMBLE GRAD-CAM ANALYSIS")
        print("="*70)
        print(f"\nImage: {Path(image_path).name}")
        print(f"\nEfficientNet prediction: {self.class_names[class_eff]} ({pred_eff[0][class_eff]*100:.2f}%)")
        print(f"Vision Transformer prediction: {self.class_names[class_vit]} ({pred_vit[0][class_vit]*100:.2f}%)")
        print(f"Ensemble prediction: {self.class_names[class_ensemble]} ({pred_ensemble[0][class_ensemble]*100:.2f}%)")
        
        # Visualize EfficientNet Grad-CAM
        if self.gradcam_eff:
            save_path = save_dir / f'gradcam_efficientnet_{Path(image_path).stem}.png'
            self.gradcam_eff.visualize(image_path, class_idx=class_ensemble, save_path=save_path)
        
        # Create comparison visualization
        self._create_comparison(
            original_img_resized,
            pred_eff[0], pred_vit[0], pred_ensemble[0],
            image_path,
            save_dir
        )
        
        print("="*70)
    
    def _create_comparison(self, original_img, pred_eff, pred_vit, pred_ensemble, 
                          image_path, save_dir):
        """Create comparison visualization of ensemble predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # EfficientNet predictions
        top_3_eff = np.argsort(pred_eff)[-3:][::-1]
        axes[0, 1].barh(range(3), [pred_eff[i]*100 for i in top_3_eff], color='#3498db')
        axes[0, 1].set_yticks(range(3))
        axes[0, 1].set_yticklabels([self.class_names[i].replace('Tomato_', '').replace('_', ' ')[:20] 
                                     for i in top_3_eff])
        axes[0, 1].set_xlabel('Confidence (%)')
        axes[0, 1].set_title('EfficientNet-B3', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlim([0, 100])
        
        # Vision Transformer predictions
        top_3_vit = np.argsort(pred_vit)[-3:][::-1]
        axes[1, 0].barh(range(3), [pred_vit[i]*100 for i in top_3_vit], color='#e74c3c')
        axes[1, 0].set_yticks(range(3))
        axes[1, 0].set_yticklabels([self.class_names[i].replace('Tomato_', '').replace('_', ' ')[:20] 
                                     for i in top_3_vit])
        axes[1, 0].set_xlabel('Confidence (%)')
        axes[1, 0].set_title('Vision Transformer', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlim([0, 100])
        
        # Ensemble predictions
        top_3_ens = np.argsort(pred_ensemble)[-3:][::-1]
        axes[1, 1].barh(range(3), [pred_ensemble[i]*100 for i in top_3_ens], color='#2ecc71')
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_yticklabels([self.class_names[i].replace('Tomato_', '').replace('_', ' ')[:20] 
                                     for i in top_3_ens])
        axes[1, 1].set_xlabel('Confidence (%)')
        axes[1, 1].set_title('Ensemble (Combined)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlim([0, 100])
        
        plt.tight_layout()
        
        save_path = save_dir / f'ensemble_comparison_{Path(image_path).stem}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Ensemble comparison saved to {save_path}")
        plt.close()


def visualize_multiple_images(model, image_folder, output_dir='results/gradcam', 
                              class_names=None, num_images=5):
    """
    Create Grad-CAM visualizations for multiple images
    
    Args:
        model: Trained model
        image_folder: Folder containing images
        output_dir: Output directory for visualizations
        class_names: List of class names
        num_images: Number of images to visualize
    """
    from glob import glob
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model)
    
    # Get image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(glob(str(Path(image_folder) / ext)))
    
    image_files = image_files[:num_images]
    
    print(f"\n📸 Generating Grad-CAM for {len(image_files)} images...")
    
    for img_path in image_files:
        img_name = Path(img_path).stem
        save_path = output_dir / f'gradcam_{img_name}.png'
        
        try:
            gradcam.visualize(img_path, save_path=save_path)
            print(f"✅ {img_name}")
        except Exception as e:
            print(f"❌ {img_name}: {e}")
    
    print(f"\n✅ Grad-CAM visualizations saved to {output_dir}")


# Main CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Grad-CAM Visual Explanations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image Grad-CAM
  python gradcam.py --model models/research/efficientnet_b3_tomato.h5 --image leaf.jpg
  
  # Ensemble Grad-CAM
  python gradcam.py --ensemble --efficientnet models/research/efficientnet_b3_tomato.h5 \\
                                --vit models/research/vision_transformer_tomato.h5 \\
                                --image leaf.jpg
  
  # Batch processing
  python gradcam.py --model models/research/efficientnet_b3_tomato.h5 --folder images/
        """
    )
    
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--efficientnet', type=str, help='Path to EfficientNet model (for ensemble)')
    parser.add_argument('--vit', type=str, help='Path to ViT model (for ensemble)')
    parser.add_argument('--image', type=str, help='Path to single image')
    parser.add_argument('--folder', type=str, help='Path to folder of images')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble Grad-CAM')
    parser.add_argument('--output', type=str, default='results/gradcam', help='Output directory')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to process (batch mode)')
    
    args = parser.parse_args()
    
    # Install opencv-python if not available
    try:
        import cv2
    except ImportError:
        print("⚠️  OpenCV not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'opencv-python'])
        import cv2
    
    if args.ensemble and args.efficientnet and args.vit and args.image:
        # Ensemble mode
        from tensorflow.keras.models import load_model
        
        eff_model = load_model(args.efficientnet)
        vit_model = load_model(args.vit)
        
        class_names = [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
            'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
            'Tomato_healthy'
        ]
        
        ensemble_gradcam = EnsembleGradCAM(eff_model, vit_model, class_names)
        ensemble_gradcam.visualize_ensemble(args.image, save_dir=args.output)
    
    elif args.model and args.image:
        # Single model, single image
        from tensorflow.keras.models import load_model
        
        model = load_model(args.model)
        gradcam = GradCAM(model)
        
        save_path = Path(args.output) / f'gradcam_{Path(args.image).stem}.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        gradcam.visualize(args.image, save_path=save_path)
    
    elif args.model and args.folder:
        # Batch mode
        from tensorflow.keras.models import load_model
        
        model = load_model(args.model)
        visualize_multiple_images(model, args.folder, args.output, num_images=args.num_images)
    
    else:
        parser.print_help()
