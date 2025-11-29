#!/usr/bin/env python3
"""
Training Data Validation CLI Tool
Provides detailed validation reports for ML training datasets.

Usage:
    python scripts/validate_training_data.py --dataset path/to/dataset
    python scripts/validate_training_data.py --dataset data/kitti/images_real --classes toy_car,toy_f1
    python scripts/validate_training_data.py --dataset data --output validation_report.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.data_validator import TrainingDataValidator, ValidationResult
from src.utils.logger import get_logger
from src.settings.settings import get_settings


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Validate training data for ML models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate dataset with auto-detected classes
  python scripts/validate_training_data.py --dataset data/kitti/images_real
  
  # Validate specific classes
  python scripts/validate_training_data.py --dataset data/kitti/images_real --classes toy_car,toy_f1
  
  # Save detailed report
  python scripts/validate_training_data.py --dataset data --output report.json
  
  # Quick validation (relaxed thresholds)
  python scripts/validate_training_data.py --dataset data --quick
  
  # Strict validation for production
  python scripts/validate_training_data.py --dataset data --strict
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    
    parser.add_argument(
        '--classes',
        type=str,
        help='Comma-separated list of class directory names (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save JSON validation report'
    )
    
    parser.add_argument(
        '--min-samples',
        type=int,
        default=100,
        help='Minimum samples per class (default: 100)'
    )
    
    parser.add_argument(
        '--min-width',
        type=int,
        default=64,
        help='Minimum image width (default: 64)'
    )
    
    parser.add_argument(
        '--min-height',
        type=int,
        default=64,
        help='Minimum image height (default: 64)'
    )
    
    parser.add_argument(
        '--max-imbalance',
        type=float,
        default=10.0,
        help='Maximum class imbalance ratio (default: 10.0)'
    )
    
    parser.add_argument(
        '--no-duplicates',
        action='store_true',
        help='Skip duplicate detection (faster)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick validation with relaxed thresholds'
    )
    
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Strict validation for production (higher thresholds)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def print_detailed_report(result: ValidationResult, verbose: bool = False):
    """Print detailed validation report to console"""
    print("\n" + "="*70)
    print(" " * 20 + "TRAINING DATA VALIDATION REPORT")
    print("="*70)
    
    # Overall Status
    status_icon = "✅" if result.is_valid else "❌"
    print(f"\n{status_icon} Overall Status: {'VALID' if result.is_valid else 'INVALID'}")
    print(f"{'='*70}\n")
    
    # Dataset Summary
    print("📊 Dataset Summary:")
    print(f"   Total Samples: {result.total_samples}")
    print(f"   Valid Samples: {result.valid_samples}")
    print(f"   Invalid Samples: {len(result.invalid_samples)}")
    
    if result.total_samples > 0:
        valid_percentage = (result.valid_samples / result.total_samples) * 100
        print(f"   Validity Rate: {valid_percentage:.1f}%")
    
    # Class Distribution
    if result.class_distribution:
        print("\n📈 Class Distribution:")
        total = sum(result.class_distribution.values())
        for class_name, count in sorted(result.class_distribution.items()):
            percentage = (count / total * 100) if total > 0 else 0
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            print(f"   {class_name:20s}: {count:5d} ({percentage:5.1f}%) {bar}")
    
    # Quality Metrics
    if result.quality_metrics:
        print("\n🔍 Quality Metrics:")
        for metric, value in result.quality_metrics.items():
            metric_name = metric.replace('_', ' ').title()
            print(f"   {metric_name:25s}: {value:8.2f}")
    
    # Warnings
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"   {i}. {warning}")
    
    # Errors
    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for i, error in enumerate(result.errors, 1):
            print(f"   {i}. {error}")
    
    # Invalid Samples
    if verbose and result.invalid_samples:
        print(f"\n🚫 Invalid Samples ({len(result.invalid_samples)}):")
        for i, sample in enumerate(result.invalid_samples[:10], 1):  # Show first 10
            print(f"   {i}. {sample}")
        if len(result.invalid_samples) > 10:
            print(f"   ... and {len(result.invalid_samples) - 10} more")
    
    # Duplicates
    if result.duplicate_files:
        print(f"\n🔄 Duplicate Files Found: {len(result.duplicate_files)}")
        if verbose:
            for i, (file1, file2) in enumerate(result.duplicate_files[:5], 1):
                print(f"   {i}. {Path(file1).name} ≈ {Path(file2).name}")
            if len(result.duplicate_files) > 5:
                print(f"   ... and {len(result.duplicate_files) - 5} more duplicates")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if result.is_valid:
        print("   ✅ Dataset is ready for training!")
        if result.warnings:
            print("   ⚠️  Address warnings for optimal results")
    else:
        print("   ❌ Fix errors before training:")
        for error in result.errors[:3]:
            print(f"      • {error}")
        if len(result.errors) > 3:
            print(f"      • ... and {len(result.errors) - 3} more issues")
    
    print("\n" + "="*70 + "\n")


def main():
    """Main CLI function"""
    args = parse_arguments()
    
    # Setup logger
    logger = get_logger(__name__)
    
    # Adjust thresholds based on mode
    if args.quick:
        min_samples = 10
        min_width = 32
        min_height = 32
        max_imbalance = 20.0
        logger.info("🚀 Quick validation mode (relaxed thresholds)")
    elif args.strict:
        min_samples = 200
        min_width = 128
        min_height = 128
        max_imbalance = 5.0
        logger.info("🔒 Strict validation mode (production thresholds)")
    else:
        min_samples = args.min_samples
        min_width = args.min_width
        min_height = args.min_height
        max_imbalance = args.max_imbalance
        logger.info("📋 Standard validation mode")
    
    # Parse class names
    class_dirs = None
    if args.classes:
        class_dirs = [c.strip() for c in args.classes.split(',')]
        logger.info(f"Validating classes: {', '.join(class_dirs)}")
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        print(f"\n❌ Error: Dataset path not found: {dataset_path}")
        return 1
    
    # Create validator
    validator = TrainingDataValidator(
        min_samples_per_class=min_samples,
        min_image_width=min_width,
        min_image_height=min_height,
        max_class_imbalance=max_imbalance,
        check_duplicates=not args.no_duplicates
    )
    
    # Run validation
    print(f"\n🔍 Validating dataset: {dataset_path}")
    print(f"⚙️  Min samples: {min_samples}, Min size: {min_width}x{min_height}")
    print(f"⚙️  Max imbalance: {max_imbalance}:1, Check duplicates: {not args.no_duplicates}")
    print("\nProcessing...")
    
    try:
        result = validator.validate_dataset(str(dataset_path), class_dirs)
        
        # Print detailed report
        print_detailed_report(result, verbose=args.verbose)
        
        # Save JSON report if requested
        if args.output:
            output_path = Path(args.output)
            result.save_report(str(output_path))
            logger.info(f"💾 Validation report saved to: {output_path}")
            print(f"💾 Detailed report saved to: {output_path}")
        
        # Return appropriate exit code
        if result.is_valid:
            logger.info("✅ Validation passed")
            return 0
        else:
            logger.warning("⚠️  Validation completed with errors")
            return 1
            
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}", exc_info=True)
        print(f"\n❌ Validation failed: {e}")
        return 2


if __name__ == '__main__':
    sys.exit(main())