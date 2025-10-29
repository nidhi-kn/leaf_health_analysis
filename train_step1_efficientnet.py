"""
STEP 1: Train EfficientNet-B3 Model
Train the first model, get results, then proceed to next step
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime


class EfficientNetTrainer:
    """Train EfficientNet-B3 step by step with visualizations"""
    
    def __init__(self, data_dir='data/tomato_health'):
        self.data_dir = Path(data_dir)
        self.img_size = (224, 224)
        self.batch_size = 16
        
        # Create output directories
        self.model_dir = Path('models/research')
        self.results_dir = Path('results/step1_efficientnet')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*70)
        print("STEP 1: TRAIN EFFICIENTNET-B3")
        print("="*70)
        
        # Enable mixed precision for faster training
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled (faster training)")
        except:
            print("ℹ️  Mixed precision not available")
    
    def prepare_data(self):
        """Load and prepare dataset"""
        print("\n📊 Loading dataset...")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # No augmentation for validation/test
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
        
        print(f"\n✅ Dataset loaded:")
        print(f"   Train: {self.train_gen.samples} images")
        print(f"   Val:   {self.val_gen.samples} images")
        print(f"   Test:  {self.test_gen.samples} images")
        print(f"   Classes: {self.num_classes}")
    
    def build_model(self):
        """Build EfficientNet-B3 model"""
        print("\n🏗️  Building EfficientNet-B3 model...")
        
        # Load pre-trained EfficientNet-B3
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
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
        
        self.model = models.Model(inputs, outputs, name='EfficientNet-B3')
        
        print(f"✅ Model built: {self.model.count_params():,} parameters")
        
        return self.model
    
    def train_stage1(self, epochs=20):
        """Stage 1: Train with frozen base"""
        print("\n" + "="*70)
        print("STAGE 1: TRAINING WITH FROZEN BASE")
        print(f"Epochs: {epochs}")
        print("="*70)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        # Callbacks
        stage1_callbacks = [
            callbacks.ModelCheckpoint(
                str(self.model_dir / 'efficientnet_stage1_best.h5'),
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
            ),
            callbacks.CSVLogger(str(self.results_dir / 'stage1_training_log.csv'))
        ]
        
        # Train
        history1 = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            callbacks=stage1_callbacks,
            verbose=1
        )
        
        print("\n✅ Stage 1 complete!")
        return history1
    
    def train_stage2(self, epochs=15):
        """Stage 2: Fine-tune top layers"""
        print("\n" + "="*70)
        print("STAGE 2: FINE-TUNING TOP LAYERS")
        print(f"Epochs: {epochs}")
        print("="*70)
        
        # Unfreeze top 20% of base model
        base_model = self.model.layers[2]
        base_model.trainable = True
        
        total_layers = len(base_model.layers)
        freeze_until = int(total_layers * 0.8)
        
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        
        trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"   Unfrozen layers: {trainable_layers}/{total_layers}")
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
        )
        
        # Callbacks
        stage2_callbacks = [
            callbacks.ModelCheckpoint(
                str(self.model_dir / 'efficientnet_stage2_best.h5'),
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
            ),
            callbacks.CSVLogger(str(self.results_dir / 'stage2_training_log.csv'))
        ]
        
        # Train
        history2 = self.model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            callbacks=stage2_callbacks,
            verbose=1
        )
        
        print("\n✅ Stage 2 complete!")
        return history2
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET")
        print("="*70)
        
        # Get predictions
        test_results = self.model.evaluate(self.test_gen, verbose=1)
        
        print("\n📊 Test Results:")
        print(f"   Loss: {test_results[0]:.4f}")
        print(f"   Accuracy: {test_results[1]*100:.2f}%")
        print(f"   Top-3 Accuracy: {test_results[2]*100:.2f}%")
        
        # Get detailed predictions
        predictions = self.model.predict(self.test_gen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_gen.classes
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            digits=4
        )
        
        print("\n📈 Classification Report:")
        print(report)
        
        # Save report
        with open(self.results_dir / 'classification_report.txt', 'w') as f:
            f.write("EfficientNet-B3 Classification Report\n")
            f.write("="*70 + "\n\n")
            f.write(f"Test Accuracy: {test_results[1]*100:.2f}%\n")
            f.write(f"Test Top-3 Accuracy: {test_results[2]*100:.2f}%\n\n")
            f.write(report)
        
        return test_results, y_pred, y_true
    
    def plot_training_history(self, history1, history2):
        """Plot training curves"""
        print("\n📊 Generating training plots...")
        
        # Combine histories
        combined_acc = history1.history['accuracy'] + history2.history['accuracy']
        combined_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
        combined_loss = history1.history['loss'] + history2.history['loss']
        combined_val_loss = history1.history['val_loss'] + history2.history['val_loss']
        
        epochs_range = range(1, len(combined_acc) + 1)
        stage1_end = len(history1.history['accuracy'])
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(epochs_range, combined_acc, 'b-', label='Training Accuracy', linewidth=2)
        axes[0].plot(epochs_range, combined_val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0].axvline(x=stage1_end, color='gray', linestyle='--', label='Fine-tuning starts')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('EfficientNet-B3 Training Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(epochs_range, combined_loss, 'b-', label='Training Loss', linewidth=2)
        axes[1].plot(epochs_range, combined_val_loss, 'r-', label='Validation Loss', linewidth=2)
        axes[1].axvline(x=stage1_end, color='gray', linestyle='--', label='Fine-tuning starts')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('EfficientNet-B3 Training Loss', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.results_dir / 'training_history.png'}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        print("\n📊 Generating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=[name.replace('Tomato_', '').replace('_', ' ')[:15] 
                               for name in self.class_names],
                   yticklabels=[name.replace('Tomato_', '').replace('_', ' ')[:15] 
                               for name in self.class_names],
                   ax=ax, cbar_kws={'label': 'Normalized Count'})
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('EfficientNet-B3 Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.results_dir / 'confusion_matrix.png'}")
        plt.close()
    
    def save_final_model(self):
        """Save the final trained model"""
        model_path = self.model_dir / 'efficientnet_b3_tomato.h5'
        self.model.save(model_path)
        
        print("\n" + "="*70)
        print("💾 MODEL SAVED")
        print("="*70)
        print(f"\n📁 Location: {model_path.absolute()}")
        print(f"   Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
        print(f"\n   How to load:")
        print(f"   >>> import tensorflow as tf")
        print(f"   >>> model = tf.keras.models.load_model('{model_path}')")
        
        return model_path


def main():
    """Main training pipeline for Step 1"""
    
    print("\n" + "="*70)
    print("🚀 STARTING EFFICIENTNET-B3 TRAINING")
    print("="*70)
    
    start_time = datetime.now()
    
    # Initialize trainer
    trainer = EfficientNetTrainer('data/tomato_health')
    
    # Step 1: Prepare data
    trainer.prepare_data()
    
    # Step 2: Build model
    trainer.build_model()
    
    # Step 3: Train Stage 1 (frozen base)
    history1 = trainer.train_stage1(epochs=5)
    
    # Step 4: Train Stage 2 (fine-tuning)
    history2 = trainer.train_stage2(epochs=5)
    
    # Step 5: Evaluate
    test_results, y_pred, y_true = trainer.evaluate_model()
    
    # Step 6: Generate visualizations
    trainer.plot_training_history(history1, history2)
    trainer.plot_confusion_matrix(y_true, y_pred)
    
    # Step 7: Save model
    model_path = trainer.save_final_model()
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*70)
    print("✅ STEP 1 COMPLETE!")
    print("="*70)
    print(f"\n⏱️  Total time: {duration}")
    print(f"\n📊 Results:")
    print(f"   Accuracy: {test_results[1]*100:.2f}%")
    print(f"   Top-3 Accuracy: {test_results[2]*100:.2f}%")
    
    print(f"\n📁 Generated Files:")
    print(f"   1. Model: {model_path}")
    print(f"   2. Training curves: results/step1_efficientnet/training_history.png")
    print(f"   3. Confusion matrix: results/step1_efficientnet/confusion_matrix.png")
    print(f"   4. Classification report: results/step1_efficientnet/classification_report.txt")
    
    print(f"\n🎯 Next Step:")
    print(f"   Run: python train_step2_vit.py")
    print("="*70)


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
    
    main()
