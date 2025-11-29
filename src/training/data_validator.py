#!/usr/bin/env python3
"""
Comprehensive Training Data Validator
Validates dataset completeness, quality, and suitability for ML training.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
from collections import Counter
import hashlib
from src.utils.logger import get_logger


@dataclass
class ImageQualityReport:
    """Report for individual image quality validation"""
    is_valid: bool
    width: int
    height: int
    channels: int
    format: str
    file_size: int
    is_corrupted: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class BalanceReport:
    """Report for class balance analysis"""
    is_balanced: bool
    class_distribution: Dict[str, int]
    imbalance_ratio: float
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AnnotationReport:
    """Report for annotation validation"""
    is_valid: bool
    total_annotations: int
    valid_annotations: int
    invalid_annotations: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Complete validation result for a dataset"""
    is_valid: bool
    total_samples: int
    valid_samples: int
    invalid_samples: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    duplicate_files: List[Tuple[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'is_valid': self.is_valid,
            'total_samples': self.total_samples,
            'valid_samples': self.valid_samples,
            'invalid_samples': self.invalid_samples,
            'warnings': self.warnings,
            'errors': self.errors,
            'class_distribution': self.class_distribution,
            'quality_metrics': self.quality_metrics,
            'duplicate_count': len(self.duplicate_files)
        }
    
    def save_report(self, output_path: str):
        """Save validation report to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            "=" * 60,
            "Training Data Validation Report",
            "=" * 60,
            f"Total Samples: {self.total_samples}",
            f"Valid Samples: {self.valid_samples}",
            f"Invalid Samples: {len(self.invalid_samples)}",
            f"Overall Status: {'✅ VALID' if self.is_valid else '❌ INVALID'}",
            "",
            "Class Distribution:",
        ]
        
        for class_name, count in self.class_distribution.items():
            percentage = (count / self.total_samples * 100) if self.total_samples > 0 else 0
            lines.append(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")
        
        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  ❌ {error}")
        
        if self.quality_metrics:
            lines.append("\nQuality Metrics:")
            for metric, value in self.quality_metrics.items():
                lines.append(f"  {metric}: {value:.2f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class TrainingDataValidator:
    """
    Comprehensive validator for training datasets.
    Checks data quality, completeness, balance, and suitability for ML training.
    """
    
    def __init__(self, 
                 min_samples_per_class: int = 100,
                 min_image_width: int = 64,
                 min_image_height: int = 64,
                 max_class_imbalance: float = 10.0,
                 allowed_formats: List[str] = None,
                 check_duplicates: bool = True):
        """
        Initialize validator with quality thresholds.
        
        Args:
            min_samples_per_class: Minimum samples required per class
            min_image_width: Minimum acceptable image width
            min_image_height: Minimum acceptable image height
            max_class_imbalance: Maximum allowed class imbalance ratio
            allowed_formats: List of allowed image formats
            check_duplicates: Whether to check for duplicate images
        """
        self.logger = get_logger(__name__)
        self.min_samples_per_class = min_samples_per_class
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.max_class_imbalance = max_class_imbalance
        self.allowed_formats = allowed_formats or ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        self.check_duplicates = check_duplicates
        
        self.logger.info(f"TrainingDataValidator initialized with min_samples={min_samples_per_class}, "
                        f"min_size={min_image_width}x{min_image_height}")
    
    def validate_dataset(self, dataset_path: str, class_dirs: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate entire dataset with comprehensive checks.
        
        Args:
            dataset_path: Root path to dataset
            class_dirs: Optional list of class directory names. If None, auto-detect.
            
        Returns:
            ValidationResult with complete validation information
        """
        self.logger.info(f"Starting dataset validation: {dataset_path}")
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            return ValidationResult(
                is_valid=False,
                total_samples=0,
                valid_samples=0,
                errors=[f"Dataset path does not exist: {dataset_path}"]
            )
        
        # Auto-detect class directories if not provided
        if class_dirs is None:
            class_dirs = self._detect_class_directories(dataset_path)
            if not class_dirs:
                self.logger.warning("No class directories detected, scanning root directory")
                class_dirs = ['.']
        
        # Initialize result
        result = ValidationResult(
            is_valid=True,
            total_samples=0,
            valid_samples=0
        )
        
        # Collect all image files
        image_files = {}
        file_hashes = {}  # For duplicate detection
        
        for class_name in class_dirs:
            class_path = dataset_path / class_name if class_name != '.' else dataset_path
            if not class_path.exists():
                result.warnings.append(f"Class directory not found: {class_name}")
                continue
            
            files = self._find_image_files(class_path)
            image_files[class_name] = files
            result.class_distribution[class_name] = len(files)
            result.total_samples += len(files)
            
            self.logger.info(f"Found {len(files)} images in class '{class_name}'")
        
        if result.total_samples == 0:
            result.is_valid = False
            result.errors.append("No image files found in dataset")
            return result
        
        # Validate minimum samples per class
        if not self.check_minimum_samples(result.class_distribution, self.min_samples_per_class):
            result.errors.append(
                f"Insufficient samples: minimum {self.min_samples_per_class} per class required"
            )
            result.is_valid = False
        
        # Validate class balance
        balance_report = self.check_class_balance(result.class_distribution, self.max_class_imbalance)
        if not balance_report.is_balanced:
            result.warnings.extend(balance_report.warnings)
            result.warnings.extend(balance_report.recommendations)
        
        # Validate individual images
        self.logger.info("Validating individual images...")
        valid_count = 0
        corrupted_count = 0
        format_errors = 0
        size_errors = 0
        
        total_width = 0
        total_height = 0
        total_file_size = 0
        
        for class_name, files in image_files.items():
            for i, img_path in enumerate(files):
                if (i + 1) % 100 == 0:
                    self.logger.info(f"  Validated {i + 1}/{len(files)} images in {class_name}")
                
                quality_report = self.validate_image_quality(str(img_path))
                
                if quality_report.is_valid:
                    valid_count += 1
                    total_width += quality_report.width
                    total_height += quality_report.height
                    total_file_size += quality_report.file_size
                    
                    # Check for duplicates
                    if self.check_duplicates:
                        file_hash = self._compute_file_hash(str(img_path))
                        if file_hash in file_hashes:
                            result.duplicate_files.append((str(img_path), file_hashes[file_hash]))
                        else:
                            file_hashes[file_hash] = str(img_path)
                else:
                    result.invalid_samples.append(str(img_path))
                    if quality_report.is_corrupted:
                        corrupted_count += 1
                    if 'Invalid format' in str(quality_report.issues):
                        format_errors += 1
                    if 'Image too small' in str(quality_report.issues):
                        size_errors += 1
        
        result.valid_samples = valid_count
        
        # Calculate quality metrics
        if valid_count > 0:
            result.quality_metrics['avg_width'] = total_width / valid_count
            result.quality_metrics['avg_height'] = total_height / valid_count
            result.quality_metrics['avg_file_size_kb'] = (total_file_size / valid_count) / 1024
            result.quality_metrics['valid_percentage'] = (valid_count / result.total_samples) * 100
        
        # Add warnings and errors
        if corrupted_count > 0:
            result.errors.append(f"Found {corrupted_count} corrupted images")
            result.is_valid = False
        
        if format_errors > 0:
            result.warnings.append(f"Found {format_errors} images with invalid format")
        
        if size_errors > 0:
            result.warnings.append(f"Found {size_errors} images below minimum size")
        
        if len(result.duplicate_files) > 0:
            result.warnings.append(f"Found {len(result.duplicate_files)} duplicate images")
        
        # Final validation check
        min_valid_percentage = 90.0  # At least 90% of images must be valid
        if result.quality_metrics.get('valid_percentage', 0) < min_valid_percentage:
            result.errors.append(
                f"Too many invalid images: {result.quality_metrics.get('valid_percentage', 0):.1f}% valid "
                f"(minimum {min_valid_percentage}% required)"
            )
            result.is_valid = False
        
        self.logger.info(f"Validation complete: {result.valid_samples}/{result.total_samples} valid samples")
        return result
    
    def check_minimum_samples(self, class_distribution: Dict[str, int], min_samples: int = 100) -> bool:
        """
        Check if dataset has minimum samples per class.
        
        Args:
            class_distribution: Dictionary of class names to sample counts
            min_samples: Minimum required samples per class
            
        Returns:
            True if all classes meet minimum requirement
        """
        for class_name, count in class_distribution.items():
            if count < min_samples:
                self.logger.warning(f"Class '{class_name}' has only {count} samples (min: {min_samples})")
                return False
        return True
    
    def validate_image_quality(self, image_path: str) -> ImageQualityReport:
        """
        Validate individual image quality and format.
        
        Args:
            image_path: Path to image file
            
        Returns:
            ImageQualityReport with validation results
        """
        path = Path(image_path)
        
        # Check file exists
        if not path.exists():
            return ImageQualityReport(
                is_valid=False,
                width=0,
                height=0,
                channels=0,
                format='',
                file_size=0,
                is_corrupted=True,
                issues=['File not found']
            )
        
        # Check format
        file_ext = path.suffix.lower().lstrip('.')
        if file_ext not in self.allowed_formats:
            return ImageQualityReport(
                is_valid=False,
                width=0,
                height=0,
                channels=0,
                format=file_ext,
                file_size=path.stat().st_size,
                is_corrupted=False,
                issues=[f'Invalid format: {file_ext} (allowed: {self.allowed_formats})']
            )
        
        # Try to load image
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return ImageQualityReport(
                    is_valid=False,
                    width=0,
                    height=0,
                    channels=0,
                    format=file_ext,
                    file_size=path.stat().st_size,
                    is_corrupted=True,
                    issues=['Failed to load image - possibly corrupted']
                )
            
            height, width = img.shape[:2]
            channels = img.shape[2] if len(img.shape) == 3 else 1
            file_size = path.stat().st_size
            
            issues = []
            warnings = []
            
            # Check dimensions
            if width < self.min_image_width or height < self.min_image_height:
                issues.append(
                    f'Image too small: {width}x{height} '
                    f'(minimum: {self.min_image_width}x{self.min_image_height})'
                )
            
            # Check file size (too small might indicate corruption)
            if file_size < 1024:  # Less than 1KB
                warnings.append('File size very small - possible corruption')
            
            # Check for blank images
            if np.std(img) < 1.0:
                warnings.append('Image appears to be blank or uniform')
            
            is_valid = len(issues) == 0
            
            return ImageQualityReport(
                is_valid=is_valid,
                width=width,
                height=height,
                channels=channels,
                format=file_ext,
                file_size=file_size,
                is_corrupted=False,
                issues=issues,
                warnings=warnings
            )
            
        except Exception as e:
            return ImageQualityReport(
                is_valid=False,
                width=0,
                height=0,
                channels=0,
                format=file_ext,
                file_size=path.stat().st_size if path.exists() else 0,
                is_corrupted=True,
                issues=[f'Error loading image: {str(e)}']
            )
    
    def check_class_balance(self, class_distribution: Dict[str, int], 
                           max_imbalance_ratio: float = 10.0) -> BalanceReport:
        """
        Check if classes are reasonably balanced.
        
        Args:
            class_distribution: Dictionary of class names to sample counts
            max_imbalance_ratio: Maximum allowed ratio between largest and smallest class
            
        Returns:
            BalanceReport with balance analysis
        """
        if not class_distribution or len(class_distribution) < 2:
            return BalanceReport(
                is_balanced=True,
                class_distribution=class_distribution,
                imbalance_ratio=1.0
            )
        
        counts = list(class_distribution.values())
        max_count = max(counts)
        min_count = min(counts)
        
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        is_balanced = imbalance_ratio <= max_imbalance_ratio
        
        warnings = []
        recommendations = []
        
        if not is_balanced:
            warnings.append(
                f'Class imbalance detected: ratio {imbalance_ratio:.2f}:1 '
                f'(max allowed: {max_imbalance_ratio}:1)'
            )
            
            # Identify underrepresented classes
            underrepresented = [
                name for name, count in class_distribution.items()
                if count < max_count / max_imbalance_ratio
            ]
            
            if underrepresented:
                recommendations.append(
                    f'Consider collecting more samples for: {", ".join(underrepresented)}'
                )
                recommendations.append(
                    'Or apply data augmentation to underrepresented classes'
                )
        
        return BalanceReport(
            is_balanced=is_balanced,
            class_distribution=class_distribution,
            imbalance_ratio=imbalance_ratio,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def validate_annotations(self, annotation_path: str) -> AnnotationReport:
        """
        Validate annotation format and completeness (for object detection datasets).
        
        Args:
            annotation_path: Path to annotation file or directory
            
        Returns:
            AnnotationReport with validation results
        """
        # Placeholder for annotation validation
        # This would be implemented based on specific annotation format (COCO, YOLO, Pascal VOC, etc.)
        return AnnotationReport(
            is_valid=True,
            total_annotations=0,
            valid_annotations=0,
            issues=['Annotation validation not yet implemented']
        )
    
    def _detect_class_directories(self, dataset_path: Path) -> List[str]:
        """Auto-detect class directories in dataset"""
        class_dirs = []
        for item in dataset_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if directory contains images
                has_images = any(
                    f.suffix.lower().lstrip('.') in self.allowed_formats
                    for f in item.iterdir()
                    if f.is_file()
                )
                if has_images:
                    class_dirs.append(item.name)
        return class_dirs
    
    def _find_image_files(self, directory: Path) -> List[Path]:
        """Recursively find all image files in directory"""
        image_files = []
        for ext in self.allowed_formats:
            image_files.extend(directory.rglob(f'*.{ext}'))
            image_files.extend(directory.rglob(f'*.{ext.upper()}'))
        return image_files
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for duplicate detection"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def validate_dataset_quick(dataset_path: str, min_samples: int = 10) -> bool:
    """
    Quick validation check for dataset (use before training).
    
    Args:
        dataset_path: Path to dataset
        min_samples: Minimum samples required
        
    Returns:
        True if dataset passes basic validation
    """
    logger = get_logger(__name__)
    validator = TrainingDataValidator(min_samples_per_class=min_samples)
    
    logger.info(f"Quick validation of dataset: {dataset_path}")
    result = validator.validate_dataset(dataset_path)
    
    if result.is_valid:
        logger.info("✅ Dataset validation passed")
        return True
    else:
        logger.error("❌ Dataset validation failed:")
        for error in result.errors:
            logger.error(f"  - {error}")
        return False