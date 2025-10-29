"""
Research Paper Quality - Multi-Model Comparison
Train and compare: EfficientNet-B3, Vision Transformer, and Ensemble

This gives you PUBLICATION-QUALITY results and analysis!
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB3
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import os
import sys

# Import config
try:
    from config import *
    from visualization import ResearchVisualizer
except ImportError:
    print(" Config or visualization module not found. Using defaults.")
    MODEL_DIR = Path('models/research')
    RESULTS_DIR = Path('results')
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16


class ResearchQualityModelTrainer:
    """
    Train multiple SOTA models for comparative research study
    
    Models:
    1. EfficientNet-B3 (SOTA CNN)
    2. Vision Transformer (Attention-based)
    3. Ensemble (Combined)
    """
    
    def __init__(self, data_dir, img_size=(224, 224), batch_size=16):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.results = {}
        self.histories = {}
        
        # Verify data directory exists
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}\n"
                           f"Please run: python prepare_data.py")
        
        # Enable mixed precision
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(" Mixed precision enabled")
        except Exception as e:
            print(f"  Mixed precision not enabled: {e}")
    
    def create_data_generators(self):
        """Create data generators"""
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Strong augmentation for better generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2]
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_gen = train_datagen.flow_from_directory(
            self.data_dir / 'train',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        self.val_gen = val_datagen.flow_from_directory(
            self.data_dir / 'val',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.test_gen = val_datagen.flow_from_directory(
            self.data_dir / 'test',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        self.num_classes = len(self.train_gen.class_indices)
        self.class_names = list(self.train_gen.class_indices.keys())
        
        print(f"\n Dataset: {self.num_classes} classes, {self.train_gen.samples} train images")
    
    # ========================================================================
    # MODEL 1: EfficientNet-B3 (State-of-the-Art CNN)
    # ========================================================================
    
    def build_efficientnet_b3(self):
        """
        EfficientNet-B3: Compound Scaling for Optimal Performance
        
        Citation: Tan & Le (2019) - "EfficientNet: Rethinking Model Scaling"
        """
        print("\n" + "="*70)
        print("BUILDING EFFICIENTNET-B3")
        print("="*70)
        
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze initially
        base_model.trainable = False
        
        # Build model
        inputs = layers.Input(shape=(*self.img_size, 3))
        x = layers.Rescaling(1./255)(inputs)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        model = models.Model(inputs, outputs, name='EfficientNet-B3')
        
        print(f" Parameters: {model.count_params():,}")
        return model
    
    def train_efficientnet_b3(self, epochs_frozen=20, epochs_finetune=15):
        """Train EfficientNet-B3 with two-stage approach"""
        print("\n Training EfficientNet-B3...")
        
        model = self.build_efficientnet_b3()
        
        # Stage 1: Frozen base
        print("\n Stage 1: Training with frozen base...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        history1 = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs_frozen,
            callbacks=self._get_callbacks('efficientnet_b3_frozen'),
            verbose=1
        )
        
        # Stage 2: Fine-tuning
        print("\n Stage 2: Fine-tuning...")
        base_model = model.layers[2]
        base_model.trainable = True
        
        # Freeze bottom 80% of layers
        for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
            layer.trainable = False
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        history2 = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs_finetune,
            callbacks=self._get_callbacks('efficientnet_b3_finetune'),
            verbose=1
        )
        
        # Evaluate
        test_results = model.evaluate(self.test_gen, verbose=1)
        
        # Save
        Path('models/research').mkdir(parents=True, exist_ok=True)
        model.save('models/research/efficientnet_b3_tomato.h5')
        
        # Combine histories
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        self.results['EfficientNet-B3'] = {
            'test_accuracy': test_results[1],
            'test_top3_accuracy': test_results[2],
            'model_path': 'models/research/efficientnet_b3_tomato.h5',
            'parameters': model.count_params(),
            'training_time': f'Stage1: {epochs_frozen}, Stage2: {epochs_finetune}'
        }
        
        self.histories['EfficientNet-B3'] = combined_history
        
        print(f"\n EfficientNet-B3: {test_results[1]*100:.2f}% accuracy")
        return model, history1, history2
    
    # ========================================================================
    # MODEL 2: Vision Transformer (Cutting-Edge)
    # ========================================================================
    
    def build_vision_transformer(self, patch_size=16, num_heads=8, transformer_layers=6):
        """
        Vision Transformer (ViT): Pure Attention Architecture
        
        Citation: Dosovitskiy et al. (2020) - "An Image is Worth 16x16 Words"
        """
        print("\n" + "="*70)
        print("BUILDING VISION TRANSFORMER")
        print("="*70)
        
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Patch extraction
        patches = self._extract_patches(inputs, patch_size)
        
        # Patch embedding
        num_patches = (self.img_size[0] // patch_size) ** 2
        projection_dim = 256
        
        patch_embeddings = layers.Dense(projection_dim)(patches)
        
        # Position embedding
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim
        )(positions)
        
        # Add position to patches
        encoded_patches = patch_embeddings + position_embedding
        
        # Transformer blocks
        for _ in range(transformer_layers):
            # Multi-head attention
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=projection_dim // num_heads,
                dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # MLP
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = layers.Dense(projection_dim * 2, activation='gelu')(x3)
            x3 = layers.Dropout(0.1)(x3)
            x3 = layers.Dense(projection_dim)(x3)
            x3 = layers.Dropout(0.1)(x3)
            encoded_patches = layers.Add()([x3, x2])
        
        # Classification head
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.GlobalAveragePooling1D()(representation)
        representation = layers.Dropout(0.3)(representation)
        
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(representation)
        
        model = models.Model(inputs, outputs, name='Vision-Transformer')
        
        print(f" Parameters: {model.count_params():,}")
        return model
    
    def _extract_patches(self, images, patch_size):
        """Extract image patches"""
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def train_vision_transformer(self, epochs=30):
        """Train Vision Transformer"""
        print("\n Training Vision Transformer...")
        
        model = self.build_vision_transformer()
        
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )
        
        history = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            callbacks=self._get_callbacks('vision_transformer'),
            verbose=1
        )
        
        # Evaluate
        test_results = model.evaluate(self.test_gen, verbose=1)
        
        # Save
        Path('models/research').mkdir(parents=True, exist_ok=True)
        model.save('models/research/vision_transformer_tomato.h5')
        
        self.results['Vision-Transformer'] = {
            'test_accuracy': test_results[1],
            'test_top3_accuracy': test_results[2],
            'model_path': 'models/research/vision_transformer_tomato.h5',
            'parameters': model.count_params(),
            'training_time': f'{epochs} epochs'
        }
        
        self.histories['Vision-Transformer'] = history.history
        
        print(f"\n Vision Transformer: {test_results[1]*100:.2f}% accuracy")
        return model, history
    
    # ========================================================================
    # MODEL 3: Ensemble (Combined Power)
    # ========================================================================
    
    def create_ensemble(self, efficientnet_model, vit_model, weights=[0.5, 0.5]):
        """
        Ensemble: Combine EfficientNet-B3 and ViT predictions
        
        Ensemble methods often outperform single models
        """
        print("\n" + "="*70)
        print("CREATING ENSEMBLE MODEL")
        print("="*70)
        
        class EnsembleModel:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict(self, x, verbose=0):
                predictions = [
                    model.predict(x, verbose=verbose) * weight
                    for model, weight in zip(self.models, self.weights)
                ]
                return np.sum(predictions, axis=0)
        
        ensemble = EnsembleModel([efficientnet_model, vit_model], weights)
        
        # Evaluate ensemble
        test_predictions = ensemble.predict(self.test_gen)
        test_labels = self.test_gen.classes
        test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == test_labels)
        
        self.results['Ensemble'] = {
            'test_accuracy': test_accuracy,
            'weights': weights,
            'components': ['EfficientNet-B3', 'Vision-Transformer']
        }
        
        print(f"\n Ensemble: {test_accuracy*100:.2f}% accuracy")
        return ensemble
    
    def _get_callbacks(self, name):
        """Get training callbacks"""
        Path('models/research/checkpoints').mkdir(parents=True, exist_ok=True)
        
        return [
            callbacks.ModelCheckpoint(
                f'models/research/checkpoints/{name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def generate_research_report(self):
        """Generate comprehensive comparison report for paper"""
        print("\n" + "="*70)
        print("GENERATING RESEARCH REPORT")
        print("="*70)
        
        # Create comparison table
        df = pd.DataFrame(self.results).T
        df = df.sort_values('test_accuracy', ascending=False)
        
        print("\n Model Comparison:")
        print(df.to_string())
        
        # Save to CSV
        Path('models/research').mkdir(parents=True, exist_ok=True)
        df.to_csv('models/research/model_comparison.csv')
        print("\n Saved to: models/research/model_comparison.csv")
        
        # Create visualization
        self._plot_comparison()
        
        # Generate LaTeX table for paper
        self._generate_latex_table(df)
        
        # Generate visualizations using ResearchVisualizer
        try:
            viz = ResearchVisualizer()
            viz.create_results_summary(self.results, self.class_names)
        except Exception as e:
            print(f"  Could not create additional visualizations: {e}")
        
        return df
    
    def _plot_comparison(self):
        """Plot model comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['test_accuracy']*100 for m in models]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.barh(models, accuracies, color=colors[:len(models)])
        
        ax.set_xlabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        
        # Dynamic x-axis limits
        min_acc = min(accuracies)
        ax.set_xlim([max(min_acc - 5, 0), 100])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}%',
                   ha='left', va='center', fontweight='bold')
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        Path('models/research').mkdir(parents=True, exist_ok=True)
        plt.savefig('models/research/model_comparison.png', dpi=300, bbox_inches='tight')
        print(" Comparison plot saved")
        plt.close()
    
    def _generate_latex_table(self, df):
        """Generate LaTeX table for paper"""
        # Create a clean version of the dataframe for LaTeX
        clean_df = df[['test_accuracy', 'parameters']].copy()
        clean_df['test_accuracy'] = clean_df['test_accuracy'] * 100
        clean_df.columns = ['Accuracy (%)', 'Parameters']
        
        latex = clean_df.to_latex(
            float_format="%.2f",
            caption="Performance comparison of deep learning models for tomato disease classification",
            label="tab:model_comparison"
        )
        
        Path('models/research').mkdir(parents=True, exist_ok=True)
        with open('models/research/table_for_paper.tex', 'w') as f:
            f.write(latex)
        
        print(" LaTeX table saved for paper")


# ============================================================================
# MAIN RESEARCH TRAINING PIPELINE
# ============================================================================

def run_research_study(data_dir='data/tomato_health'):
    """
    Run complete research study with multiple models
    
    This gives you publication-quality results!
    """
    print("="*70)
    print("RESEARCH-QUALITY MULTI-MODEL STUDY")
    print("Train 3 Models + Generate Comparative Analysis")
    print("="*70)
    
    trainer = ResearchQualityModelTrainer(data_dir)
    trainer.create_data_generators()
    
    # Train Model 1: EfficientNet-B3
    print("\n" + "="*70)
    print("TRAINING MODEL 1/3: EFFICIENTNET-B3")
    print("="*70)
    efficientnet, hist_eff1, hist_eff2 = trainer.train_efficientnet_b3()
    
    # Train Model 2: Vision Transformer
    print("\n" + "="*70)
    print("TRAINING MODEL 2/3: VISION TRANSFORMER")
    print("="*70)
    vit, hist_vit = trainer.train_vision_transformer()
    
    # Create Model 3: Ensemble
    print("\n" + "="*70)
    print("CREATING MODEL 3/3: ENSEMBLE")
    print("="*70)
    ensemble = trainer.create_ensemble(efficientnet, vit)
    
    # Generate research report
    comparison_df = trainer.generate_research_report()
    
    # Generate Grad-CAM visualizations
    print("\n" + "="*70)
    print("GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*70)
    try:
        from gradcam import EnsembleGradCAM
        import random
        
        # Get sample test images (5 random images)
        test_images = []
        test_dir = Path(data_dir) / 'test'
        for class_folder in test_dir.iterdir():
            if class_folder.is_dir():
                images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
                if images:
                    test_images.append(random.choice(images))
        
        if test_images[:5]:
            gradcam = EnsembleGradCAM(efficientnet, vit, trainer.class_names)
            for img_path in test_images[:5]:
                try:
                    gradcam.visualize_ensemble(str(img_path))
                except Exception as e:
                    print(f"⚠️  Could not generate Grad-CAM for {img_path.name}: {e}")
            
            print("✅ Grad-CAM visualizations generated!")
        else:
            print("⚠️  No test images found for Grad-CAM")
    except Exception as e:
        print(f"⚠️  Grad-CAM generation skipped: {e}")
    
    print("\n" + "="*70)
    print("✅ RESEARCH STUDY COMPLETE!")
    print("="*70)
    
    print("\n📁 Generated Files:")
    print("  1. models/research/efficientnet_b3_tomato.h5")
    print("  2. models/research/vision_transformer_tomato.h5")
    print("  3. models/research/model_comparison.csv")
    print("  4. models/research/model_comparison.png")
    print("  5. models/research/table_for_paper.tex")
    print("  6. results/gradcam/ - Visual explanations (Grad-CAM heatmaps)")
    
    print("\n📝 For Your Research Paper:")
    print("  ✅ Use the comparison table (LaTeX)")
    print("  ✅ Include the performance plot")
    print("  ✅ Include Grad-CAM visualizations (shows WHERE model looks)")
    print("  ✅ Discuss why ensemble > single model")
    print("  ✅ Compare CNN vs Transformer architecture")
    
    return trainer, comparison_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/tomato_health')
    args = parser.parse_args()
    
    # Enable GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Run complete research study
    trainer, results = run_research_study(args.data_dir)
    
    print("\n🎉 Ready for research paper submission!")