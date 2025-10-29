"""
Model Evaluation and Prediction Script
Use this after training to evaluate models or make predictions on new images
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import json


class TomatoDiseasePredictor:
    """
    Load trained models and make predictions
    """
    
    def __init__(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path: Path to saved model (.h5 file)
        """
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Load class names
        self.class_names = [
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
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to image file
            target_size: Target image size
        
        Returns:
            Preprocessed image array
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, img
    
    def predict(self, image_path, top_k=3):
        """
        Make prediction on a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = {
            'predictions': [],
            'image_path': str(image_path)
        }
        
        for idx in top_indices:
            results['predictions'].append({
                'disease': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'confidence_percent': float(predictions[idx] * 100)
            })
        
        return results, original_img, predictions
    
    def predict_and_visualize(self, image_path, save_path=None):
        """
        Predict and visualize results
        
        Args:
            image_path: Path to image file
            save_path: Optional path to save visualization
        """
        results, original_img, all_predictions = self.predict(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Show image
        ax1.imshow(original_img)
        ax1.axis('off')
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        
        # Show predictions
        top_diseases = [p['disease'].replace('Tomato_', '').replace('_', ' ') 
                       for p in results['predictions']]
        top_confidences = [p['confidence_percent'] for p in results['predictions']]
        
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_diseases))]
        bars = ax2.barh(top_diseases, top_confidences, color=colors)
        
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 100])
        
        # Add value labels
        for bar, conf in zip(bars, top_confidences):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{conf:.2f}%',
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Print results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"\nImage: {Path(image_path).name}")
        print("\nTop Predictions:")
        for i, pred in enumerate(results['predictions'], 1):
            print(f"  {i}. {pred['disease']:<45} {pred['confidence_percent']:>6.2f}%")
        print("="*70)
        
        return results


def evaluate_on_test_set(model_path, test_data_dir):
    """
    Evaluate model on entire test set
    
    Args:
        model_path: Path to saved model
        test_data_dir: Path to test data directory
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    print("\n📊 Evaluating...")
    results = model.evaluate(test_gen, verbose=1)
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]*100:.2f}%")
    if len(results) > 2:
        print(f"Top-3 Accuracy: {results[2]*100:.2f}%")
    print("="*70)
    
    return results


def compare_models(model_paths, image_path):
    """
    Compare predictions from multiple models
    
    Args:
        model_paths: List of model paths
        image_path: Path to image to predict
    """
    print("\n" + "="*70)
    print("COMPARING MULTIPLE MODELS")
    print("="*70)
    
    results_all = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\n🔍 {model_name}...")
        
        predictor = TomatoDiseasePredictor(model_path)
        results, _, _ = predictor.predict(image_path, top_k=1)
        
        results_all[model_name] = results['predictions'][0]
    
    # Display comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(f"\nImage: {Path(image_path).name}\n")
    
    for model_name, pred in results_all.items():
        print(f"{model_name:<30} → {pred['disease']:<40} ({pred['confidence_percent']:.2f}%)")
    
    print("="*70)
    
    return results_all


def batch_predict(model_path, image_folder, output_csv='predictions.csv'):
    """
    Make predictions on a folder of images
    
    Args:
        model_path: Path to saved model
        image_folder: Path to folder containing images
        output_csv: Output CSV file path
    """
    import pandas as pd
    
    print("\n" + "="*70)
    print("BATCH PREDICTION")
    print("="*70)
    
    predictor = TomatoDiseasePredictor(model_path)
    
    # Get all images
    image_folder = Path(image_folder)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(list(image_folder.glob(ext)))
    
    print(f"\n📁 Found {len(image_files)} images")
    
    # Predict on each image
    results_list = []
    for img_path in image_files:
        try:
            results, _, _ = predictor.predict(img_path, top_k=1)
            pred = results['predictions'][0]
            
            results_list.append({
                'filename': img_path.name,
                'predicted_disease': pred['disease'],
                'confidence': pred['confidence_percent']
            })
            
            print(f"✅ {img_path.name:<40} → {pred['disease']:<40} ({pred['confidence_percent']:.1f}%)")
        
        except Exception as e:
            print(f"❌ {img_path.name}: {e}")
    
    # Save to CSV
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Results saved to {output_csv}")
    print("="*70)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and make predictions with trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on single image
  python evaluate.py --model models/research/efficientnet_b3_tomato.h5 --image path/to/image.jpg
  
  # Evaluate on test set
  python evaluate.py --model models/research/efficientnet_b3_tomato.h5 --evaluate --test_dir data/tomato_health/test
  
  # Compare multiple models
  python evaluate.py --compare --models models/research/*.h5 --image path/to/image.jpg
  
  # Batch predict on folder
  python evaluate.py --model models/research/efficientnet_b3_tomato.h5 --batch --folder path/to/images/
        """
    )
    
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--models', type=str, nargs='+', help='Paths to multiple models for comparison')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on test set')
    parser.add_argument('--test_dir', type=str, default='data/tomato_health/test',
                       help='Path to test data directory')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--batch', action='store_true', help='Batch predict on folder')
    parser.add_argument('--folder', type=str, help='Folder containing images for batch prediction')
    parser.add_argument('--save', type=str, help='Path to save visualization')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file for batch predictions')
    
    args = parser.parse_args()
    
    # Mode 1: Single prediction
    if args.image and args.model and not args.compare and not args.batch:
        predictor = TomatoDiseasePredictor(args.model)
        predictor.predict_and_visualize(args.image, save_path=args.save)
    
    # Mode 2: Evaluate on test set
    elif args.evaluate and args.model:
        evaluate_on_test_set(args.model, args.test_dir)
    
    # Mode 3: Compare models
    elif args.compare and args.models and args.image:
        compare_models(args.models, args.image)
    
    # Mode 4: Batch prediction
    elif args.batch and args.model and args.folder:
        batch_predict(args.model, args.folder, args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
