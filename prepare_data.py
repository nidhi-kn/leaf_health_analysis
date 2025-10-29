"""
Data Preparation Script for Tomato Disease Classification
Splits PlantVillage dataset into train/val/test sets
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

def prepare_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test sets
    
    Args:
        source_dir: Path to PlantVillage folder
        output_dir: Path to output directory
        train_ratio: Proportion for training set (default: 0.7)
        val_ratio: Proportion for validation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    print("="*70)
    print("PREPARING TOMATO DISEASE DATASET")
    print("="*70)
    print(f"\nSource: {source_path}")
    print(f"Output: {output_path}")
    print(f"\nSplit Ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}")
    print(f"\nFound {len(class_dirs)} disease classes:")
    
    total_images = 0
    class_stats = {}
    
    # Process each class
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        
        # Get all images in this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
            image_files.extend(list(class_dir.glob(ext)))
        
        if not image_files:
            print(f"WARNING: Skipping {class_name} - no images found")
            continue
        
        random.shuffle(image_files)
        total_images += len(image_files)
        
        # Calculate split indices
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split files
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        class_stats[class_name] = {
            'total': n_total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
        
        # Copy files to respective directories
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_class_dir = output_path / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in files:
                dest = split_class_dir / img_file.name
                if not dest.exists():
                    shutil.copy2(img_file, dest)
    
    # Print statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\n{'Class':<50} {'Total':<8} {'Train':<8} {'Val':<8} {'Test':<8}")
    print("-"*82)
    
    total_train, total_val, total_test = 0, 0, 0
    for class_name, stats in class_stats.items():
        print(f"{class_name:<50} {stats['total']:<8} {stats['train']:<8} {stats['val']:<8} {stats['test']:<8}")
        total_train += stats['train']
        total_val += stats['val']
        total_test += stats['test']
    
    print("-"*82)
    print(f"{'TOTAL':<50} {total_images:<8} {total_train:<8} {total_val:<8} {total_test:<8}")
    
    print("\nDataset preparation complete!")
    print(f"Data saved to: {output_path}")
    
    # Save statistics to file
    stats_file = output_path / 'dataset_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("TOMATO DISEASE DATASET STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Number of Classes: {len(class_stats)}\n\n")
        f.write(f"Train: {total_train} images ({total_train/total_images*100:.1f}%)\n")
        f.write(f"Val:   {total_val} images ({total_val/total_images*100:.1f}%)\n")
        f.write(f"Test:  {total_test} images ({total_test/total_images*100:.1f}%)\n\n")
        f.write("Class Distribution:\n")
        f.write("-"*70 + "\n")
        for class_name, stats in class_stats.items():
            f.write(f"\n{class_name}:\n")
            f.write(f"  Total: {stats['total']}\n")
            f.write(f"  Train: {stats['train']}\n")
            f.write(f"  Val:   {stats['val']}\n")
            f.write(f"  Test:  {stats['test']}\n")
    
    print(f"Statistics saved to: {stats_file}")
    
    return class_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare tomato disease dataset')
    parser.add_argument('--source', type=str, 
                       default='dataset/PlantVillage',
                       help='Path to source PlantVillage directory')
    parser.add_argument('--output', type=str, 
                       default='data/tomato_health',
                       help='Path to output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Prepare dataset
    stats = prepare_dataset(
        args.source, 
        args.output,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    print("\nReady to train models!")
    print("   Run: python main.py --data_dir data/tomato_health")
