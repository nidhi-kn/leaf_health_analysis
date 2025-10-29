"""
Visualization Utilities for Research Paper
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResearchVisualizer:
    """Create publication-quality visualizations"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, histories, model_names, save_path=None):
        """
        Plot training history for multiple models
        
        Args:
            histories: List of training history dictionaries
            model_names: List of model names
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, (history, name) in enumerate(zip(histories, model_names)):
            color = colors[idx % len(colors)]
            
            # Plot accuracy
            if 'accuracy' in history.history:
                axes[0].plot(history.history['accuracy'], 
                           label=f'{name} (Train)', 
                           color=color, linewidth=2)
            if 'val_accuracy' in history.history:
                axes[0].plot(history.history['val_accuracy'], 
                           label=f'{name} (Val)', 
                           color=color, linestyle='--', linewidth=2)
            
            # Plot loss
            if 'loss' in history.history:
                axes[1].plot(history.history['loss'], 
                           label=f'{name} (Train)', 
                           color=color, linewidth=2)
            if 'val_loss' in history.history:
                axes[1].plot(history.history['val_loss'], 
                           label=f'{name} (Val)', 
                           color=color, linestyle='--', linewidth=2)
        
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Model Loss Comparison', fontsize=14, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history saved to {save_path}")
        else:
            plt.savefig(self.results_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, model_name, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Customize ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               title=f'Confusion Matrix - {model_name}',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2f})',
                       ha="center", va="center",
                       color="white" if cm_normalized[i, j] > thresh else "black",
                       fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        else:
            plt.savefig(self.results_dir / f'confusion_matrix_{model_name.lower()}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_model_comparison(self, results_dict, save_path=None):
        """
        Plot bar chart comparing model performances
        
        Args:
            results_dict: Dictionary with model results
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(results_dict.keys())
        accuracies = [results_dict[m]['test_accuracy'] * 100 for m in models]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.barh(models, accuracies, color=colors[:len(models)], height=0.6)
        
        ax.set_xlabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xlim([min(accuracies) - 2, 100])
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{acc:.2f}%',
                   ha='left', va='center', fontweight='bold', fontsize=12)
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Model comparison saved to {save_path}")
        else:
            plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_classification_report(self, y_true, y_pred, class_names, model_name):
        """
        Generate and save classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            model_name: Name of the model
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=class_names, 
                                      digits=4)
        
        report_path = self.results_dir / f'classification_report_{model_name.lower()}.txt'
        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        
        print(f"✅ Classification report saved to {report_path}")
        
        # Also create a DataFrame version
        report_dict = classification_report(y_true, y_pred, 
                                           target_names=class_names, 
                                           output_dict=True)
        df = pd.DataFrame(report_dict).transpose()
        df.to_csv(self.results_dir / f'classification_report_{model_name.lower()}.csv')
        
        return report
    
    def plot_class_distribution(self, data_generator, save_path=None):
        """
        Plot class distribution
        
        Args:
            data_generator: Keras data generator
            save_path: Path to save the plot
        """
        class_counts = {}
        for class_name, class_idx in data_generator.class_indices.items():
            count = np.sum(data_generator.classes == class_idx)
            class_counts[class_name] = count
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        bars = ax.bar(range(len(classes)), counts, color='#3498db', alpha=0.7)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Class distribution saved to {save_path}")
        else:
            plt.savefig(self.results_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_results_summary(self, results_dict, class_names, save_path=None):
        """
        Create a comprehensive results summary for the paper
        
        Args:
            results_dict: Dictionary with all model results
            class_names: List of class names
            save_path: Path to save the summary
        """
        summary_path = save_path or self.results_dir / 'results_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RESEARCH RESULTS SUMMARY\n")
            f.write("Multi-Model Deep Learning for Tomato Disease Classification\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: {len(class_names)} disease classes\n\n")
            
            f.write("Model Performance:\n")
            f.write("-"*70 + "\n")
            
            for model_name, results in results_dict.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Test Accuracy: {results['test_accuracy']*100:.2f}%\n")
                if 'test_top3_accuracy' in results:
                    f.write(f"  Top-3 Accuracy: {results['test_top3_accuracy']*100:.2f}%\n")
                if 'parameters' in results:
                    f.write(f"  Parameters: {results['parameters']:,}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Key Findings:\n")
            f.write("-"*70 + "\n")
            
            # Find best model
            best_model = max(results_dict.items(), key=lambda x: x[1]['test_accuracy'])
            f.write(f"\n✅ Best Model: {best_model[0]} ({best_model[1]['test_accuracy']*100:.2f}%)\n")
            
            # Calculate improvement
            if 'Ensemble' in results_dict and 'EfficientNet-B3' in results_dict:
                improvement = (results_dict['Ensemble']['test_accuracy'] - 
                             results_dict['EfficientNet-B3']['test_accuracy']) * 100
                f.write(f"✅ Ensemble improvement over EfficientNet: +{improvement:.2f}%\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"✅ Results summary saved to {summary_path}")


def visualize_sample_predictions(model, test_gen, class_names, num_samples=16, save_path='results/sample_predictions.png'):
    """
    Visualize sample predictions
    
    Args:
        model: Trained model
        test_gen: Test data generator
        class_names: List of class names
        num_samples: Number of samples to show
        save_path: Path to save the plot
    """
    # Get a batch of test images
    x_batch, y_batch = next(test_gen)
    predictions = model.predict(x_batch[:num_samples], verbose=0)
    
    # Plot
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        
        # Display image
        ax.imshow(x_batch[idx])
        
        # Get true and predicted labels
        true_label = class_names[np.argmax(y_batch[idx])]
        pred_label = class_names[np.argmax(predictions[idx])]
        confidence = np.max(predictions[idx]) * 100
        
        # Set title with color based on correctness
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                    fontsize=8, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Sample predictions saved to {save_path}")
    plt.close()
