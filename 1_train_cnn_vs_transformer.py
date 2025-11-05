"""
CNN vs Transformer Comparison Study
MobileNetV2 vs Vision Transformer for Tomato Disease Classification

This is the main training script for comparing:
1. MobileNetV2 (Efficient CNN for mobile deployment)
2. Vision Transformer (Attention-based architecture)  
3. Ensemble (Combined approach)

Usage:
    python 1_train_cnn_vs_transformer.py

Expected Results:
    - MobileNetV2: 90-95% accuracy
    - Vision Transformer: 92-97% accuracy
    - Ensemble: 95-98% accuracy
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime
import math


class CNNvsTransformerComparison:
    """Compare CNN (MobileNetV2) vs Transformer (ViT) architectures"""
    
    def __init__(self, data_dir='data/tomato_health'):
        self.data_dir = Path(data_dir)
        self.img_size = (224, 224)
        self.batch_size = 32  # Increased batch size for faster training
        
        # Create output directories
        self.model_dir = Path('models/research')
        self.results_dir = Path('results/cnn_vs_transformer')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("CNN vs TRANSFORMER COMPARISON STUDY")
        print("MobileNetV2 vs Vision Transformer")
        print("="*70)
        
        # Enable mixed precision
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled")
        except:
            print("ℹ️  Mixed precision not available")
        
        # Store results for comparison
        self.results = {}
        self.histories = {}
    
    def prepare_data(self):
        """Prepare dataset with advanced augmentation"""
        print("\n📊 Preparing dataset...")
        
        # Advanced data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.4,
            height_shift_range=0.4,
            shear_range=0.3,
            zoom_range=0.4,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load data
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
        
        # Calculate class weights
        self.class_weights = self._calculate_class_weights()
        
        print(f"\n✅ Dataset prepared:")
        print(f"   Train: {self.train_gen.samples} images")
        print(f"   Val:   {self.val_gen.samples} images")
        print(f"   Test:  {self.test_gen.samples} images")
        print(f"   Classes: {self.num_classes}")
    
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        class_counts = {}
        for class_name, class_idx in self.train_gen.class_indices.items():
            count = np.sum(self.train_gen.classes == class_idx)
            class_counts[class_idx] = count
        
        total_samples = sum(class_counts.values())
        class_weights = {}
        
        for class_idx, count in class_counts.items():
            class_weights[class_idx] = total_samples / (self.num_classes * count)
        
        return class_weights
    
    # ========================================================================
    # MODEL 1: MobileNetV2 (CNN Architecture)
    # ========================================================================
    
    def build_mobilenetv2(self):
        """
        Build MobileNetV2 model
        
        MobileNetV2 Features:
        - Depthwise separable convolutions
        - Inverted residual blocks
        - Linear bottlenecks
        - ~3.5M parameters
        - Optimized for mobile deployment
        """
        print("\n🏗️  Building MobileNetV2 (CNN)...")
        
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # MobileNetV2 base (using the approach that worked in diagnostic)
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_tensor=inputs,
            pooling='avg'  # Use built-in global average pooling
        )
        
        # Freeze base initially
        base_model.trainable = False
        
        # Simple but effective classification head
        x = base_model.output
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs, name='MobileNetV2')
        
        print(f"✅ MobileNetV2 built: {model.count_params():,} parameters")
        return model
    
    # ========================================================================
    # MODEL 2: Vision Transformer (Attention Architecture)
    # ========================================================================
    
    def build_vision_transformer(self, patch_size=16, num_heads=6, transformer_layers=4):
        """
        Build Vision Transformer model (Optimized with better initialization)
        
        Vision Transformer Features:
        - Pure attention mechanism (no convolutions)
        - Patch-based image processing
        - Multi-head self-attention
        - Global context modeling
        - Better initialization for small datasets
        """
        print("\n🏗️  Building Vision Transformer (Optimized)...")
        
        inputs = layers.Input(shape=(*self.img_size, 3))
        
        # Add data augmentation within model for better generalization
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        
        # Create patches
        patches = self._extract_patches(x, patch_size)
        
        # Patch embedding with better initialization
        num_patches = (self.img_size[0] // patch_size) ** 2
        projection_dim = 256  # Increased back for better capacity
        
        # Linear projection of patches with better initialization
        patch_embeddings = layers.Dense(
            projection_dim,
            kernel_initializer='he_normal'
        )(patches)
        
        # Add positional embeddings with better initialization
        positions = tf.range(start=0, limit=num_patches, delta=1)
        position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim,
            embeddings_initializer='normal'
        )(positions)
        
        # Add position to patch embeddings with layer norm
        encoded_patches = patch_embeddings + position_embedding
        encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Transformer encoder blocks
        for i in range(transformer_layers):
            # Multi-head self-attention
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=projection_dim // num_heads,
                dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            
            # MLP block
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
        
        # Enhanced classification head with residual connection
        x = layers.Dense(512, activation='gelu', kernel_initializer='he_normal')(representation)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='gelu', kernel_initializer='he_normal')(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        model = models.Model(inputs, outputs, name='Vision-Transformer')
        
        print(f"✅ Vision Transformer built: {model.count_params():,} parameters")
        return model
    
    def _extract_patches(self, images, patch_size):
        """Extract image patches for Vision Transformer using Keras layers"""
        # Use Lambda layer to wrap TensorFlow operations
        def extract_patches_fn(x):
            batch_size = tf.shape(x)[0]
            patches = tf.image.extract_patches(
                images=x,
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, patch_size, patch_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )
            patch_dims = patches.shape[-1]
            patches = tf.reshape(patches, [batch_size, -1, patch_dims])
            return patches
        
        return layers.Lambda(extract_patches_fn)(images)
    
    # ========================================================================
    # TRAINING FUNCTIONS
    # ========================================================================
    
    def train_mobilenetv2(self, epochs_frozen=15, epochs_finetune=10):
        """Train MobileNetV2 with two-stage approach"""
        print(f"\n" + "="*70)
        print("TRAINING MODEL 1/2: MOBILENETV2 (CNN)")
        print("="*70)
        
        model = self.build_mobilenetv2()
        
        # Stage 1: Frozen base
        print(f"\n🔒 Stage 1: Frozen base training ({epochs_frozen} epochs)")
        
        # Base model is already frozen in build_mobilenetv2()
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),  # Same LR that worked in diagnostic
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        stage1_callbacks = [
            callbacks.ModelCheckpoint(
                str(self.model_dir / 'mobilenetv2_stage1_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(str(self.results_dir / 'mobilenetv2_stage1_log.csv'))
        ]
        
        history1 = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs_frozen,
            callbacks=stage1_callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Stage 2: Fine-tuning
        print(f"\n🔓 Stage 2: Fine-tuning ({epochs_finetune} epochs)")
        
        # Unfreeze MobileNetV2 base model
        base_model = model.layers[1]  # MobileNetV2 base
        base_model.trainable = True
        
        # Freeze bottom 70% of layers for stable fine-tuning
        total_layers = len(base_model.layers)
        freeze_until = int(total_layers * 0.7)
        
        for i, layer in enumerate(base_model.layers):
            if i < freeze_until:
                layer.trainable = False
        
        trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"   Unfrozen: {trainable_count}/{total_layers} layers")
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        stage2_callbacks = [
            callbacks.ModelCheckpoint(
                str(self.model_dir / 'mobilenetv2_final.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(str(self.results_dir / 'mobilenetv2_stage2_log.csv'))
        ]
        
        history2 = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs_finetune,
            callbacks=stage2_callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Combine histories
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        # Evaluate
        test_results = model.evaluate(self.test_gen, verbose=1)
        
        # Store results
        self.results['MobileNetV2'] = {
            'test_accuracy': test_results[1],
            'test_top3_accuracy': test_results[2],
            'parameters': model.count_params(),
            'architecture': 'CNN (Convolutional Neural Network)',
            'key_features': 'Depthwise separable convolutions, inverted residuals, mobile-optimized'
        }
        
        self.histories['MobileNetV2'] = combined_history
        
        print(f"\n✅ MobileNetV2 training complete!")
        print(f"   Test Accuracy: {test_results[1]*100:.2f}%")
        print(f"   Top-3 Accuracy: {test_results[2]*100:.2f}%")
        
        return model
    
    def load_existing_mobilenet(self, model_path):
        """Load existing trained MobileNet model"""
        print(f"\n🔄 Loading existing MobileNet model from: {model_path}")
        
        try:
            model = tf.keras.models.load_model(model_path)
            print("✅ MobileNet model loaded successfully!")
            
            # Evaluate the loaded model to get current performance
            print("📊 Evaluating loaded model...")
            test_results = model.evaluate(self.test_gen, verbose=1)
            
            # Store results
            self.results['MobileNetV2'] = {
                'test_accuracy': test_results[1],
                'test_top3_accuracy': test_results[2] if len(test_results) > 2 else 0.0,
                'parameters': model.count_params(),
                'architecture': 'CNN (Convolutional Neural Network)',
                'key_features': 'Depthwise separable convolutions, inverted residuals, mobile-optimized',
                'note': 'Pre-trained model (loaded from existing)'
            }
            
            print(f"\n✅ MobileNetV2 loaded successfully!")
            print(f"   Test Accuracy: {test_results[1]*100:.2f}%")
            if len(test_results) > 2:
                print(f"   Top-3 Accuracy: {test_results[2]*100:.2f}%")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading MobileNet model: {e}")
            print("🔄 Falling back to training new model...")
            return self.train_mobilenetv2(epochs_frozen=15, epochs_finetune=10)
    
    def train_vision_transformer(self, epochs=10):
        """Train Vision Transformer with optimized settings and learning schedule"""
        print(f"\n" + "="*70)
        print("TRAINING MODEL 2/2: VISION TRANSFORMER (OPTIMIZED)")
        print("="*70)
        
        model = self.build_vision_transformer()
        
        # Optimized compilation with cosine decay learning rate
        initial_lr = 1e-3
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=epochs * len(self.train_gen),
            alpha=0.1
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(
                learning_rate=lr_schedule, 
                weight_decay=1e-4,
                beta_1=0.9,
                beta_2=0.999
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        vit_callbacks = [
            callbacks.ModelCheckpoint(
                str(self.model_dir / 'vision_transformer_final.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # Reduced patience for faster stopping
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,  # More aggressive LR reduction
                patience=3,  # Faster LR reduction
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.CSVLogger(str(self.results_dir / 'vision_transformer_log.csv'))
        ]
        
        history = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            callbacks=vit_callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        
        # Evaluate
        test_results = model.evaluate(self.test_gen, verbose=1)
        
        # Store results
        self.results['Vision-Transformer'] = {
            'test_accuracy': test_results[1],
            'test_top3_accuracy': test_results[2],
            'parameters': model.count_params(),
            'architecture': 'Transformer (Attention-based)',
            'key_features': 'Self-attention, patch embeddings, global context'
        }
        
        self.histories['Vision-Transformer'] = history.history
        
        print(f"\n✅ Vision Transformer training complete!")
        print(f"   Test Accuracy: {test_results[1]*100:.2f}%")
        print(f"   Top-3 Accuracy: {test_results[2]*100:.2f}%")
        
        return model
    
    def create_ensemble(self, mobilenet_model, vit_model, weights=[0.5, 0.5]):
        """Create ensemble combining CNN and Transformer"""
        print(f"\n" + "="*70)
        print("CREATING ENSEMBLE MODEL (CNN + TRANSFORMER)")
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
        
        ensemble = EnsembleModel([mobilenet_model, vit_model], weights)
        
        # Evaluate ensemble
        print("📊 Evaluating ensemble...")
        test_predictions = ensemble.predict(self.test_gen)
        test_labels = self.test_gen.classes
        test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == test_labels)
        
        # Top-3 accuracy
        top3_accuracy = np.mean([
            test_labels[i] in np.argsort(test_predictions[i])[-3:]
            for i in range(len(test_labels))
        ])
        
        self.results['Ensemble'] = {
            'test_accuracy': test_accuracy,
            'test_top3_accuracy': top3_accuracy,
            'architecture': 'Ensemble (CNN + Transformer)',
            'key_features': 'Combines MobileNet efficiency with Transformer global attention',
            'components': ['MobileNetV2', 'Vision-Transformer'],
            'weights': weights
        }
        
        print(f"\n✅ Ensemble created!")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"   Top-3 Accuracy: {top3_accuracy*100:.2f}%")
        
        return ensemble
    
    def generate_research_report(self):
        """Generate comprehensive research comparison report"""
        print(f"\n" + "="*70)
        print("GENERATING RESEARCH COMPARISON REPORT")
        print("="*70)
        
        # Create comparison DataFrame
        df = pd.DataFrame(self.results).T
        df = df.sort_values('test_accuracy', ascending=False)
        
        print("\n📊 CNN vs Transformer Comparison Results:")
        print(df[['test_accuracy', 'test_top3_accuracy', 'parameters', 'architecture']].to_string())
        
        # Save detailed results
        df.to_csv(self.results_dir / 'cnn_vs_transformer_comparison.csv')
        print(f"\n✅ Results saved to: {self.results_dir / 'cnn_vs_transformer_comparison.csv'}")
        
        # Generate visualizations
        self._plot_architecture_comparison()
        self._plot_training_histories()
        self._generate_latex_table(df)
        self._generate_research_insights()
        
        return df
    
    def _plot_architecture_comparison(self):
        """Plot comprehensive architecture comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['test_accuracy']*100 for m in models]
        top3_accs = [self.results[m]['test_top3_accuracy']*100 for m in models]
        params = [self.results[m].get('parameters', 0)/1e6 for m in models if 'parameters' in self.results[m]]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        # Test accuracy comparison
        bars1 = axes[0, 0].bar(models, accuracies, color=colors[:len(models)], alpha=0.8)
        axes[0, 0].set_ylabel('Test Accuracy (%)', fontsize=12)
        axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([min(accuracies) - 5, 100])
        
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Top-3 accuracy comparison
        bars2 = axes[0, 1].bar(models, top3_accs, color=colors[:len(models)], alpha=0.8)
        axes[0, 1].set_ylabel('Top-3 Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Top-3 Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim([min(top3_accs) - 2, 100])
        
        for bar, acc in zip(bars2, top3_accs):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                           f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Parameter comparison (excluding ensemble)
        model_names = [m for m in models if 'parameters' in self.results[m]]
        if len(params) >= 2:
            bars3 = axes[1, 0].bar(model_names, params[:len(model_names)], 
                                  color=colors[:len(model_names)], alpha=0.8)
            axes[1, 0].set_ylabel('Parameters (Millions)', fontsize=12)
            axes[1, 0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
            
            for bar, param in zip(bars3, params[:len(model_names)]):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                               f'{param:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Architecture type visualization
        arch_types = ['CNN', 'Transformer', 'Ensemble']
        arch_colors = ['#3498db', '#e74c3c', '#2ecc71']
        axes[1, 1].pie([1, 1, 1], labels=arch_types, colors=arch_colors, autopct='',
                      startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1, 1].set_title('Architecture Types Compared', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'cnn_vs_transformer_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved: {self.results_dir / 'cnn_vs_transformer_comparison.png'}")
        plt.close()
    
    def _plot_training_histories(self):
        """Plot training histories for both models"""
        if len(self.histories) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['#3498db', '#e74c3c']
        model_names = ['MobileNetV2', 'Vision-Transformer']
        
        for idx, (model_name, color) in enumerate(zip(model_names, colors)):
            if model_name in self.histories:
                history = self.histories[model_name]
                epochs = range(1, len(history['accuracy']) + 1)
                
                # Accuracy plots
                axes[0, idx].plot(epochs, history['accuracy'], color=color, linewidth=2, label='Training')
                axes[0, idx].plot(epochs, history['val_accuracy'], color=color, linewidth=2, linestyle='--', label='Validation')
                axes[0, idx].set_title(f'{model_name} - Accuracy', fontweight='bold')
                axes[0, idx].set_xlabel('Epoch')
                axes[0, idx].set_ylabel('Accuracy')
                axes[0, idx].legend()
                axes[0, idx].grid(True, alpha=0.3)
                
                # Loss plots
                axes[1, idx].plot(epochs, history['loss'], color=color, linewidth=2, label='Training')
                axes[1, idx].plot(epochs, history['val_loss'], color=color, linewidth=2, linestyle='--', label='Validation')
                axes[1, idx].set_title(f'{model_name} - Loss', fontweight='bold')
                axes[1, idx].set_xlabel('Epoch')
                axes[1, idx].set_ylabel('Loss')
                axes[1, idx].legend()
                axes[1, idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_histories.png', dpi=300, bbox_inches='tight')
        print(f"✅ Training histories saved: {self.results_dir / 'training_histories.png'}")
        plt.close()
    
    def _generate_latex_table(self, df):
        """Generate LaTeX table for research paper"""
        # Clean DataFrame for LaTeX
        clean_df = df[['test_accuracy', 'test_top3_accuracy', 'parameters', 'architecture']].copy()
        clean_df['test_accuracy'] = clean_df['test_accuracy'] * 100
        clean_df['test_top3_accuracy'] = clean_df['test_top3_accuracy'] * 100
        clean_df.columns = ['Test Accuracy (%)', 'Top-3 Accuracy (%)', 'Parameters', 'Architecture']
        
        latex = clean_df.to_latex(
            float_format="%.2f",
            caption="Performance comparison of CNN and Transformer architectures for tomato disease classification",
            label="tab:cnn_vs_transformer_comparison"
        )
        
        with open(self.results_dir / 'comparison_table.tex', 'w') as f:
            f.write(latex)
        
        print(f"✅ LaTeX table saved: {self.results_dir / 'comparison_table.tex'}")
    
    def _generate_research_insights(self):
        """Generate research insights and analysis"""
        insights_path = self.results_dir / 'research_insights.txt'
        
        with open(insights_path, 'w') as f:
            f.write("CNN vs TRANSFORMER RESEARCH INSIGHTS\n")
            f.write("="*70 + "\n\n")
            
            # Performance analysis
            cnn_acc = self.results['MobileNetV2']['test_accuracy']
            vit_acc = self.results['Vision-Transformer']['test_accuracy']
            ensemble_acc = self.results['Ensemble']['test_accuracy']
            
            f.write("PERFORMANCE ANALYSIS:\n")
            f.write("-"*70 + "\n")
            f.write(f"MobileNetV2 (CNN):         {cnn_acc*100:.2f}%\n")
            f.write(f"Vision Transformer:         {vit_acc*100:.2f}%\n")
            f.write(f"Ensemble:                   {ensemble_acc*100:.2f}%\n\n")
            
            # Winner analysis
            if vit_acc > cnn_acc:
                winner = "Vision Transformer"
                improvement = (vit_acc - cnn_acc) * 100
                f.write(f"🏆 WINNER: {winner} (+{improvement:.2f}% over CNN)\n\n")
            else:
                winner = "MobileNetV2"
                improvement = (cnn_acc - vit_acc) * 100
                f.write(f"🏆 WINNER: {winner} (+{improvement:.2f}% over Transformer)\n\n")
            
            # Ensemble analysis
            best_individual = max(cnn_acc, vit_acc)
            ensemble_improvement = (ensemble_acc - best_individual) * 100
            f.write(f"📈 ENSEMBLE IMPROVEMENT: +{ensemble_improvement:.2f}% over best individual model\n\n")
            
            # Architecture insights
            f.write("ARCHITECTURE INSIGHTS:\n")
            f.write("-"*70 + "\n")
            f.write("CNN (MobileNetV2) Strengths:\n")
            f.write("• Local feature extraction through convolutions\n")
            f.write("• Depthwise separable convolutions for efficiency\n")
            f.write("• Inverted residual blocks with linear bottlenecks\n")
            f.write("• Mobile-optimized architecture with fewer parameters\n")
            f.write("• Strong inductive biases for image data\n\n")
            
            f.write("Transformer (ViT) Strengths:\n")
            f.write("• Global context modeling through self-attention\n")
            f.write("• No architectural inductive biases\n")
            f.write("• Scalable to larger datasets\n")
            f.write("• Captures long-range dependencies\n\n")
            
            f.write("Ensemble Benefits:\n")
            f.write("• Combines local CNN features with global Transformer attention\n")
            f.write("• Reduces individual model biases\n")
            f.write("• Improved robustness and generalization\n")
            f.write("• Best of both architectural paradigms\n\n")
            
            # Research implications
            f.write("RESEARCH IMPLICATIONS:\n")
            f.write("-"*70 + "\n")
            f.write("1. Both CNN and Transformer architectures are viable for plant disease detection\n")
            f.write("2. Ensemble methods effectively combine complementary strengths\n")
            f.write("3. Transfer learning remains effective for both architectures\n")
            f.write("4. Agricultural AI benefits from architectural diversity\n")
        
        print(f"✅ Research insights saved: {insights_path}")


def main():
    """Main comparison pipeline"""
    
    print("\n" + "="*70)
    print("🚀 CNN vs TRANSFORMER COMPARISON STUDY")
    print("="*70)
    
    start_time = datetime.now()
    
    # Initialize comparison system
    comparison = CNNvsTransformerComparison('data/tomato_health')
    
    # Prepare data
    comparison.prepare_data()
    
    # Load existing MobileNetV2 (CNN) - already trained to 86.12%
    print("\n🎯 Loading existing CNN Architecture...")
    mobilenet_model = comparison.load_existing_mobilenet('models/quick_test/mobilenet_working.h5')
    
    # Train Vision Transformer (Optimized for speed)
    print("\n🎯 Training Transformer Architecture...")
    vit_model = comparison.train_vision_transformer(epochs=10)
    
    # Create ensemble
    print("\n🎯 Creating Ensemble Model...")
    ensemble = comparison.create_ensemble(mobilenet_model, vit_model)
    
    # Generate research report
    results_df = comparison.generate_research_report()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("✅ CNN vs TRANSFORMER STUDY COMPLETE!")
    print("="*70)
    print(f"\n⏱️  Total time: {duration}")
    
    print(f"\n📊 Final Results:")
    for model_name, results in comparison.results.items():
        print(f"   {model_name:<20} {results['test_accuracy']*100:>6.2f}%")
    
    print(f"\n🏆 Research Findings:")
    cnn_acc = comparison.results['MobileNetV2']['test_accuracy']
    vit_acc = comparison.results['Vision-Transformer']['test_accuracy']
    ensemble_acc = comparison.results['Ensemble']['test_accuracy']
    
    if vit_acc > cnn_acc:
        print(f"   • Transformer outperforms CNN by {(vit_acc - cnn_acc)*100:.2f}%")
    else:
        print(f"   • CNN outperforms Transformer by {(cnn_acc - vit_acc)*100:.2f}%")
    
    best_individual = max(cnn_acc, vit_acc)
    print(f"   • Ensemble improves over best individual by {(ensemble_acc - best_individual)*100:.2f}%")
    
    print(f"\n📁 Generated Files:")
    print(f"   1. Models: models/research/")
    print(f"   2. Results: results/cnn_vs_transformer/")
    print(f"   3. Comparison plots: cnn_vs_transformer_comparison.png")
    print(f"   4. LaTeX table: comparison_table.tex")
    print(f"   5. Research insights: research_insights.txt")
    
    print(f"\n🎓 Ready for research paper submission!")
    print(f"   Title: 'CNN vs Transformer for Tomato Disease Classification'")
    
    return comparison, results_df


if __name__ == "__main__":
    # Enable GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU enabled: {len(gpus)} device(s)")
        except Exception as e:
            print(f"⚠️  GPU setup warning: {e}")
    else:
        print("ℹ️  No GPU detected, using CPU")
    
    comparison, results = main()
    
    print("\n🎉 CNN vs Transformer comparison complete!")