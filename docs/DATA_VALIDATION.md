# Training Data Validation Documentation

## Overview

The Smart Traffic Light system now includes comprehensive training data validation to ensure high-quality ML model training. This prevents training with insufficient, corrupted, or poorly balanced datasets.

## Features

### 1. **Comprehensive Data Validation**
- ✅ Minimum sample requirements per class
- ✅ Image quality checks (resolution, format, corruption)
- ✅ Class balance analysis
- ✅ Duplicate detection
- ✅ File format validation
- ✅ Image integrity verification

### 2. **Integration Points**
- **Training Scripts**: Automatic validation before training starts
- **Test Scripts**: Pre-training data quality checks
- **CLI Tool**: Standalone validation for datasets
- **Model Loading**: Runtime model availability validation

### 3. **Quality Metrics**
- Valid sample percentage
- Average image dimensions
- Average file size
- Class distribution
- Imbalance ratios

## Usage

### CLI Tool

```bash
# Basic validation
python scripts/validate_training_data.py --dataset data/kitti/images_real

# Specify classes
python scripts/validate_training_data.py --dataset data/kitti/images_real --classes toy_car,toy_f1

# Save detailed report
python scripts/validate_training_data.py --dataset data --output validation_report.json

# Quick validation (relaxed thresholds)
python scripts/validate_training_data.py --dataset data --quick

# Strict validation (production)
python scripts/validate_training_data.py --dataset data --strict

# Verbose output with details
python scripts/validate_training_data.py --dataset data --verbose
```

### Python API

```python
from src.training.data_validator import TrainingDataValidator

# Create validator
validator = TrainingDataValidator(
    min_samples_per_class=100,
    min_image_width=64,
    min_image_height=64,
    max_class_imbalance=10.0,
    check_duplicates=True
)

# Validate dataset
result = validator.validate_dataset('data/kitti/images_real')

# Check results
if result.is_valid:
    print("✅ Dataset is ready for training!")
else:
    print("❌ Issues found:")
    for error in result.errors:
        print(f"  - {error}")

# Save report
result.save_report('validation_report.json')
print(result.get_summary())
```

### Quick Validation

```python
from src.training.data_validator import validate_dataset_quick

# Quick check before training
if validate_dataset_quick('data/kitti/images_real', min_samples=50):
    print("Dataset ready!")
    # Proceed with training
else:
    print("Dataset validation failed")
    # Stop and fix issues
```

## Integration with Training Scripts

### Custom Car Trainer

```python
from src.training.custom_car_trainer import LightweightCarTrainer

trainer = LightweightCarTrainer()

# Training with validation (default)
accuracy, time = trainer.train_model(validate_data=True)

# Skip validation (not recommended)
accuracy, time = trainer.train_model(validate_data=False)
```

### Advanced Car Trainer

```python
from src.training.advanced_car_trainer import AdvancedCarTrainer

trainer = AdvancedCarTrainer()

# Validation is automatic and stricter for deep learning
model, fp32_acc, int8_acc = trainer.train_model(validate_data=True)
```

### Test Scripts

```python
# test_simple.py includes automatic validation
python test_simple.py

# Validates:
# - Data availability
# - Image quality
# - Sample counts
# - Class balance
```

## Configuration

### Environment Variables

Add to `.env` file:

```bash
# Training validation settings
MIN_SAMPLES_PER_CLASS=100
MIN_IMAGE_WIDTH=64
MIN_IMAGE_HEIGHT=64
MAX_CLASS_IMBALANCE=10.0
TRAINING_ALLOWED_FORMATS=jpg,jpeg,png,bmp,webp
CHECK_DUPLICATES=true
VALIDATE_BEFORE_TRAINING=true

# Training settings
ENABLE_DATA_AUGMENTATION=true
AUGMENTATION_FACTOR=3
TEST_SPLIT=0.2
```

### Settings API

```python
from src.settings.settings import get_settings

settings = get_settings()

# Access training settings
print(f"Min samples: {settings.training.min_samples_per_class}")
print(f"Min size: {settings.training.min_image_width}x{settings.training.min_image_height}")
print(f"Validate: {settings.training.validate_before_training}")
```

## Validation Thresholds

### Quick Mode (Development)
- Min samples per class: 10
- Min image size: 32x32
- Max class imbalance: 20:1
- **Use for**: Quick testing, development

### Standard Mode (Default)
- Min samples per class: 100
- Min image size: 64x64
- Max class imbalance: 10:1
- **Use for**: Regular training

### Strict Mode (Production)
- Min samples per class: 200
- Min image size: 128x128
- Max class imbalance: 5:1
- **Use for**: Production models, critical applications

## Validation Reports

### Console Output

```
======================================================================
                TRAINING DATA VALIDATION REPORT
======================================================================

✅ Overall Status: VALID
======================================================================

📊 Dataset Summary:
   Total Samples: 1250
   Valid Samples: 1238
   Invalid Samples: 12
   Validity Rate: 99.0%

📈 Class Distribution:
   toy_car            :   650 ( 52.0%) ██████████████████████████
   toy_f1             :   600 ( 48.0%) ████████████████████████

🔍 Quality Metrics:
   Avg Width              :   320.50
   Avg Height             :   240.30
   Avg File Size Kb       :    45.20
   Valid Percentage       :    99.04

⚠️  Warnings (2):
   1. Found 3 duplicate images
   2. 12 images below minimum size

💡 Recommendations:
   ✅ Dataset is ready for training!
   ⚠️  Address warnings for optimal results

======================================================================
```

### JSON Report

```json
{
  "is_valid": true,
  "total_samples": 1250,
  "valid_samples": 1238,
  "invalid_samples": [...],
  "warnings": [...],
  "errors": [],
  "class_distribution": {
    "toy_car": 650,
    "toy_f1": 600
  },
  "quality_metrics": {
    "avg_width": 320.5,
    "avg_height": 240.3,
    "avg_file_size_kb": 45.2,
    "valid_percentage": 99.04
  },
  "duplicate_count": 3
}
```

## Model Validation

The system also validates ML model availability at startup:

```python
from src.models.car_identify import CarIdentifier

# Models are validated on initialization
identifier = CarIdentifier(
    use_tflite=True,
    use_custom_model=True,
    use_ml=True
)

# Validation results logged:
# ✅ TFLite model validated successfully
# ✅ Custom SVM model validated successfully
# 🎯 Active models: tflite, custom_svm
```

## Best Practices

### 1. **Always Validate Before Training**
```python
# Good ✅
if validate_dataset_quick(dataset_path):
    train_model()

# Bad ❌
train_model()  # No validation
```

### 2. **Address Warnings**
Even if validation passes, address warnings for optimal results:
- Remove duplicates
- Balance classes
- Fix corrupted images
- Ensure minimum quality

### 3. **Use Appropriate Thresholds**
- Development: Quick mode
- Testing: Standard mode
- Production: Strict mode

### 4. **Regular Validation**
Run validation CLI periodically:
```bash
python scripts/validate_training_data.py --dataset data --strict --output report.json
```

### 5. **Monitor Quality Metrics**
Track metrics over time:
- Valid percentage should be > 95%
- Class imbalance should be < 10:1
- Average image size should match model requirements

## Troubleshooting

### Common Issues

#### 1. **Insufficient Samples**
```
❌ Error: Insufficient samples: minimum 100 per class required
```

**Solution**: Collect more training data or use data augmentation

#### 2. **Corrupted Images**
```
❌ Error: Found 15 corrupted images
```

**Solution**: Remove or replace corrupted files

#### 3. **Class Imbalance**
```
⚠️  Warning: Class imbalance detected: ratio 15.5:1
```

**Solution**: 
- Collect more samples for underrepresented classes
- Apply data augmentation
- Use class weights during training

#### 4. **Images Too Small**
```
⚠️  Warning: Found 25 images below minimum size
```

**Solution**: 
- Use higher resolution images
- Adjust `min_image_width` and `min_image_height` settings

## API Reference

### TrainingDataValidator

```python
class TrainingDataValidator:
    def __init__(self,
                 min_samples_per_class: int = 100,
                 min_image_width: int = 64,
                 min_image_height: int = 64,
                 max_class_imbalance: float = 10.0,
                 allowed_formats: List[str] = None,
                 check_duplicates: bool = True)
    
    def validate_dataset(self, 
                        dataset_path: str,
                        class_dirs: Optional[List[str]] = None) -> ValidationResult
    
    def validate_image_quality(self, image_path: str) -> ImageQualityReport
    
    def check_class_balance(self,
                           class_distribution: Dict[str, int],
                           max_imbalance_ratio: float = 10.0) -> BalanceReport
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool
    total_samples: int
    valid_samples: int
    invalid_samples: List[str]
    warnings: List[str]
    errors: List[str]
    class_distribution: Dict[str, int]
    quality_metrics: Dict[str, float]
    duplicate_files: List[Tuple[str, str]]
    
    def to_dict(self) -> dict
    def save_report(self, output_path: str)
    def get_summary(self) -> str
```

## Examples

### Example 1: Basic Validation

```python
from src.training import TrainingDataValidator

validator = TrainingDataValidator()
result = validator.validate_dataset('data/kitti/images_real')

if result.is_valid:
    print(f"✅ {result.valid_samples}/{result.total_samples} samples valid")
else:
    print(f"❌ Validation failed: {len(result.errors)} errors")
```

### Example 2: Custom Thresholds

```python
validator = TrainingDataValidator(
    min_samples_per_class=50,
    min_image_width=32,
    min_image_height=32,
    max_class_imbalance=15.0
)

result = validator.validate_dataset('data/custom_dataset')
```

### Example 3: Detailed Report

```python
result = validator.validate_dataset('data')

print(result.get_summary())
result.save_report('detailed_report.json')

# Access metrics
for metric, value in result.quality_metrics.items():
    print(f"{metric}: {value:.2f}")
```

## Support

For issues or questions:
1. Check validation report errors and warnings
2. Review this documentation
3. Check logs in `logs/` directory
4. Run with `--verbose` flag for detailed output

## Future Enhancements

- [ ] Annotation format validation (COCO, YOLO, Pascal VOC)
- [ ] Advanced quality metrics (blur detection, contrast analysis)
- [ ] Automatic data cleaning suggestions
- [ ] Integration with data augmentation pipeline
- [ ] Web dashboard for validation reports