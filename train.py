"""
Training Launcher Script
Easy-to-use script to prepare data and train models
"""

import os
import sys
from pathlib import Path
import argparse


def check_dependencies():
    """Check if all dependencies are installed"""
    print("="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    missing_packages = []
    required_packages = [
        'tensorflow',
        'numpy',
        'pandas',
        'sklearn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'PIL'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Missing packages detected!")
        print(f"   Install with: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def check_gpu():
    """Check GPU availability"""
    try:
        import tensorflow as tf
        print("\n" + "="*70)
        print("GPU CHECK")
        print("="*70)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower)")
            print("   This is okay but will take longer.")
            return False
    except Exception as e:
        print(f"⚠️  Error checking GPU: {e}")
        return False


def prepare_dataset(args):
    """Prepare dataset by splitting into train/val/test"""
    print("\n" + "="*70)
    print("STEP 1: PREPARING DATASET")
    print("="*70)
    
    data_dir = Path('data/tomato_health')
    
    # Check if data is already prepared
    if (data_dir / 'train').exists() and (data_dir / 'val').exists() and (data_dir / 'test').exists():
        print("✅ Dataset already prepared!")
        print(f"   Location: {data_dir}")
        
        if not args.force_prepare:
            response = input("\n   Re-prepare dataset? (y/n): ").lower()
            if response != 'y':
                return True
    
    # Prepare dataset
    print("\n📦 Preparing dataset...")
    try:
        from prepare_data import prepare_dataset
        
        stats = prepare_dataset(
            source_dir='dataset/PlantVillage',
            output_dir='data/tomato_health',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        print("\n✅ Dataset preparation complete!")
        return True
    except Exception as e:
        print(f"\n❌ Error preparing dataset: {e}")
        print("   Please check that 'dataset/PlantVillage' exists")
        return False


def train_models(args):
    """Train all models"""
    print("\n" + "="*70)
    print("STEP 2: TRAINING MODELS")
    print("="*70)
    
    try:
        import tensorflow as tf
        from main import run_research_study
        
        # Configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Run training
        print("\n🚀 Starting training...")
        print("   This will take approximately 5-6 hours")
        print("   You can safely leave this running overnight")
        print("\n" + "-"*70)
        
        trainer, results = run_research_study('data/tomato_health')
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(
        description='Train Multi-Model System for Tomato Disease Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (prepare data + train models)
  python train.py
  
  # Skip data preparation
  python train.py --skip-prepare
  
  # Force re-prepare dataset
  python train.py --force-prepare
  
  # Only prepare dataset
  python train.py --prepare-only
        """
    )
    
    parser.add_argument('--skip-prepare', action='store_true',
                       help='Skip dataset preparation')
    parser.add_argument('--force-prepare', action='store_true',
                       help='Force re-prepare dataset even if exists')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare dataset, do not train')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and GPU checks')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TOMATO DISEASE CLASSIFICATION - RESEARCH STUDY")
    print("Multi-Model Training System")
    print("="*70)
    print("\nModels to be trained:")
    print("  1. EfficientNet-B3 (SOTA CNN)")
    print("  2. Vision Transformer (Attention-based)")
    print("  3. Ensemble (Combined)")
    print("\n" + "="*70)
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            print("\n❌ Please install missing dependencies first")
            print("   Run: pip install -r requirements.txt")
            return
        
        check_gpu()
    
    # Prepare dataset
    if not args.skip_prepare:
        if not prepare_dataset(args):
            print("\n❌ Dataset preparation failed. Exiting.")
            return
        
        if args.prepare_only:
            print("\n✅ Dataset preparation complete!")
            print("   Run 'python train.py --skip-prepare' to train models")
            return
    
    # Train models
    if not train_models(args):
        print("\n❌ Training failed. Please check the error messages above.")
        return
    
    print("\n" + "="*70)
    print("🎉 SUCCESS!")
    print("="*70)
    print("\n📁 Check these files for your research paper:")
    print("   - models/research/model_comparison.csv")
    print("   - models/research/model_comparison.png")
    print("   - models/research/table_for_paper.tex")
    print("   - results/results_summary.txt")
    print("\n📝 Next steps:")
    print("   1. Review the results in models/research/")
    print("   2. Use the LaTeX table in your paper")
    print("   3. Include the comparison plots")
    print("   4. Analyze the confusion matrices")
    print("\n🎓 Good luck with your research paper!")


if __name__ == "__main__":
    main()
