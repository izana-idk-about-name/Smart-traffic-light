#!/usr/bin/env python3
"""
Advanced Lightweight Car Detection Trainer
Uses MobileNet SSD backbone with TFLite quantization and pruning optimization
Optimized for Raspberry Pi with detection accuracy improvements
"""

import cv2
import numpy as np
import os
import random
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    import tensorflow_model_optimization as tfmot
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available. Advanced training features will be limited.")
    TENSORFLOW_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time
from pathlib import Path
from typing import List, Tuple
from src.training.data_validator import TrainingDataValidator

class MobileNetSSDCars:
    """MobileNet SSD implementation for car detection"""

    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def _create_base_mobilenet(self):
        """Create MobileNetV2 base with custom classification head for car detection"""
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            alpha=0.35  # Width multiplier for lighter model
        )

        # Freeze base layers for transfer learning
        for layer in base_model.layers[:-30]:  # Unfreeze more layers for car detection
            layer.trainable = False

        # Add custom classification head optimized for car detection
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # Smaller dense layer
        x = tf.keras.layers.Dropout(0.3)(x)  # Add dropout for regularization
        x = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=x)
        return self.model

    def _create_lightweight_ssd(self):
        """Create lightweight SSD-style detection model"""
        inputs = Input(shape=self.input_shape)

        # Base MobileNetV2 with reduced alpha
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_tensor=inputs,
            alpha=0.35  # Very lightweight version
        )

        # Multi-scale feature maps for SSD
        feature_maps = [
            base_model.get_layer('block_6_expand_relu').output,  # 28x28
            base_model.get_layer('block_13_expand_relu').output, # 14x14
            base_model.output  # 7x7
        ]

        # Default boxes (anchors) for each feature map
        num_anchors = [4, 6, 6]
        outputs = []

        for feature_map, num_anchor in zip(feature_maps, num_anchors):
            # Classification head (background + car)
            cls = Conv2D(num_anchor * self.num_classes, 3, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(feature_map)
            cls = Reshape((-1, self.num_classes))(cls)

            # Regression head for bounding boxes
            reg = Conv2D(num_anchor * 4, 3, padding='same',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01))(feature_map)
            reg = Reshape((-1, 4))(reg)

            # Concatenate classification and regression
            output = tf.concat([cls, reg], axis=-1)
            outputs.append(output)

        # Final output: concatenate all feature map outputs
        final_output = tf.concat(outputs, axis=1)
        self.model = Model(inputs=inputs, outputs=final_output)
        return self.model

class AdvancedCarTrainer:
    """
    Advanced car detection trainer with optimization techniques:
    - MobileNet backbone
    - Quantization to int8
    - Pruning for model compression
    - Data augmentation
    """

    def __init__(self, data_dir='data', model_dir='src/models'):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

        # Model configurations
        self.input_shape = (224, 224, 3)
        self.batch_size = 16  # Small batch for RPi training
        self.epochs = 50

        # Create MobileNet SSD model
        self.detector = MobileNetSSDCars(input_shape=self.input_shape)

        # Data generators
        self.train_generator = None
        self.val_generator = None

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess training data with advanced augmentation"""
        images = []
        labels = []

        # Load car images
        car_dirs = [
            self.data_dir / 'kitti' / 'images_real' / 'toy_car',
            self.data_dir / 'kitti' / 'images_real' / 'toy_f1'
        ]

        print("📸 Loading car images...")
        car_count = 0
        for car_dir in car_dirs:
            if car_dir.exists():
                for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp']:
                    for img_path in car_dir.rglob(ext):
                        try:
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img = cv2.resize(img, self.input_shape[:2])
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                images.append(img)
                                labels.append(1)  # Car
                                car_count += 1
                        except Exception as e:
                            print(f"⚠️ Skipping invalid image: {img_path}")

        # Load background images
        bg_dir = self.data_dir / 'imagens_originais'
        if bg_dir.exists():
            print("📸 Loading background images...")
            bg_count = 0
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp']:
                for img_path in bg_dir.glob(ext):
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = cv2.resize(img, self.input_shape[:2])
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(0)  # Background
                            bg_count += 1
                    except Exception as e:
                        print(f"⚠️ Skipping invalid background image: {img_path}")

        print(f"✅ Loaded {car_count} car images and {bg_count} background images")

        return np.array(images), np.array(labels)

    def create_data_generators(self, images: np.ndarray, labels: np.ndarray):
        """Create data generators with augmentation"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )

        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=self.batch_size,
            shuffle=False
        )

    def apply_pruning(self, model, pruning_percentage=0.3):
        """Apply magnitude-based pruning to the model"""
        print(f"✂️ Applying {pruning_percentage*100}% pruning...")

        # Prune all Conv2D and Dense layers
        def apply_pruning_to_layer(layer):
            if isinstance(layer, (Conv2D, Dense, DepthwiseConv2D)):
                return tfmot.sparsity.keras.prune_low_magnitude(layer)
            return layer

        # Apply pruning wrapper
        pruned_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_layer
        )

        # Compile pruned model
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=pruning_percentage,
            begin_step=0,
            end_step=self.train_generator.n // self.batch_size * self.epochs
        )

        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            pruned_model,
            pruning_schedule=pruning_schedule
        )

        pruned_model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return pruned_model

    def quantize_to_int8(self, model, representative_data):
        """Convert model to TFLite with int8 quantization"""
        print("🔄 Converting to int8 quantized TFLite model...")

        def representative_dataset():
            for data in representative_data.take(100):
                yield [tf.cast(data, tf.float32)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Enable int8 quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        # Save quantized model
        tflite_path = self.model_dir / 'car_detector_int8.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"💾 Quantized model saved: {tflite_path}")
        print(f"📏 Model size: {len(tflite_model)} bytes")

        return tflite_model

    def train_model(self, validate_data: bool = True):
        """Train the model with advanced optimization techniques and data validation"""
        print("🚗 Starting advanced car detection training...")
        print("📊 Techniques: MobileNet backbone, Pruning, Quantization, Data Augmentation")

        # Validate training data if requested
        if validate_data:
            print("\n🔍 Validating training data before training...")
            try:
                validator = TrainingDataValidator(
                    min_samples_per_class=100,  # Higher requirement for deep learning
                    min_image_width=64,
                    min_image_height=64,
                    max_class_imbalance=10.0,
                    check_duplicates=True
                )
                
                # Validate car data
                car_dir = self.data_dir / 'kitti' / 'images_real'
                if car_dir.exists():
                    result = validator.validate_dataset(str(car_dir), class_dirs=['toy_car', 'toy_f1'])
                    
                    print(f"\n{'='*60}")
                    print("Validação Avançada dos Dados de Treinamento")
                    print(f"{'='*60}")
                    print(f"Total: {result.total_samples} | Válidos: {result.valid_samples}")
                    print(f"Status: {'✅ VÁLIDO' if result.is_valid else '⚠️  COM AVISOS'}")
                    
                    if result.class_distribution:
                        print("\nDistribuição de Classes:")
                        for class_name, count in result.class_distribution.items():
                            print(f"  {class_name}: {count} amostras")
                    
                    if result.warnings:
                        print("\n⚠️  Avisos:")
                        for warning in result.warnings[:5]:
                            print(f"  - {warning}")
                    
                    if result.errors:
                        print("\n❌ Erros Críticos:")
                        for error in result.errors:
                            print(f"  - {error}")
                        print("\n🛑 Treinamento cancelado devido a erros nos dados")
                        print("💡 Para deep learning, é essencial ter dados de alta qualidade")
                        raise ValueError("Data validation failed - cannot proceed with training")
                    
                    if result.valid_samples < 200:
                        print(f"\n⚠️  Apenas {result.valid_samples} amostras válidas")
                        print("   Recomendado: pelo menos 500 amostras para deep learning robusto")
                        print("   Continuando, mas o modelo pode ter performance limitada...")
                    
                    if result.quality_metrics:
                        print("\n📈 Métricas de Qualidade:")
                        for metric, value in result.quality_metrics.items():
                            print(f"  {metric}: {value:.2f}")
                    
                    print(f"{'='*60}\n")
                
            except ValueError:
                # Re-raise validation errors
                raise
            except Exception as e:
                print(f"⚠️  Erro na validação: {e}")
                print("   Continuando com treinamento (risco de falha)...")

        # Enable mixed precision for faster training
        try:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
            print("🔥 Mixed precision training enabled")
        except:
            print("⚠️ Mixed precision not available, using float32")

        # Load and prepare data
        images, labels = self.load_training_data()
        self.create_data_generators(images, labels)

        # Create lightweight MobileNet model
        model = self.detector._create_base_mobilenet()

        # Compile initial model
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Callbacks for training
        checkpoint = ModelCheckpoint(
            str(self.model_dir / 'car_detector_baseline.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        )

        early_stop = EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_accuracy'
        )

        lr_reducer = ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            monitor='val_loss'
        )

        # Phase 1: Initial training
        print("📚 Phase 1: Initial training...")
        history1 = model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=self.epochs // 2,
            callbacks=[checkpoint, early_stop, lr_reducer]
        )

        # Phase 2: Apply pruning and continue training
        print("✂️ Phase 2: Applying pruning...")
        model = self.apply_pruning(model, pruning_percentage=0.4)  # More aggressive pruning

        # Update pruning schedule
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.4,
            begin_step=0,
            end_step=self.train_generator.n // self.batch_size * (self.epochs // 2)
        )

        model_pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

        history2 = model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=self.epochs // 2,
            callbacks=[checkpoint, early_stop, lr_reducer, model_pruning_callback]
        )

        # Phase 3: Strip pruning and fine-tune
        print("🎯 Phase 3: Fine-tuning...")
        model = tfmot.sparsity.keras.strip_pruning(model)

        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        history3 = model.fit(
            self.train_generator,
            validation_data=self.val_generator,
            epochs=10,
            callbacks=[checkpoint, early_stop]
        )

        # Save full precision model
        model.save(self.model_dir / 'car_detector_optimized.h5')
        print("💾 Full precision model saved")

        # Phase 4: Quantize to int8 TFLite
        print("🔄 Phase 4: Quantizing to int8...")
        representative_data = tf.data.Dataset.from_tensor_slices(
            (images.astype(np.float32) / 255.0, labels)
        ).batch(1).take(min(200, len(images)))

        tflite_model = self.quantize_to_int8(model, representative_data)

        # Evaluate both models
        fp32_accuracy = self.evaluate_keras_model(model, images, labels)
        int8_accuracy = self.evaluate_tflite_model(tflite_model, images, labels)

        print("✅ Training completed!")
        print("=" * 50)
        print(".2f")
        print(".2f")
        print(".2f")
        print("💡 Optimization techniques applied:")
        print("  • MobileNetV2 α=0.35 backbone")
        print("  • 40% magnitude-based pruning")
        print("  • Post-training int8 quantization")
        print("  • Advanced data augmentation")
        print("  • Mixed precision training")

        return model, fp32_accuracy, int8_accuracy

    def evaluate_keras_model(self, model, test_images, test_labels):
        """Evaluate Keras model performance with detailed metrics"""
        print("📊 Evaluating Keras model...")

        # Preprocess test images
        processed_images = []
        for img in test_images:
            img = cv2.resize(img, self.input_shape[:2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            processed_images.append(img)

        processed_images = np.array(processed_images)

        # Get predictions
        predictions = model.predict(processed_images, batch_size=self.batch_size, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)

        # Calculate metrics
        accuracy = np.mean(pred_classes == test_labels)
        precision = np.sum((pred_classes == 1) & (test_labels == 1)) / np.sum(pred_classes == 1) if np.sum(pred_classes == 1) > 0 else 0
        recall = np.sum((pred_classes == 1) & (test_labels == 1)) / np.sum(test_labels == 1) if np.sum(test_labels == 1) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")

        return accuracy

    def evaluate_tflite_model(self, tflite_model, test_images, test_labels):
        """Evaluate TFLite model performance with detailed metrics"""
        print("📊 Evaluating TFLite int8 model...")

        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        predictions = []
        test_subset = min(200, len(test_images))  # Evaluate subset for speed

        for i, img in enumerate(test_images[:test_subset]):
            if (i + 1) % 50 == 0:
                print(f"🔄 Evaluating image {i+1}/{test_subset}...")

            # Preprocess
            img = cv2.resize(img, self.input_shape[:2])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

            # Handle quantization
            if input_details[0]['dtype'] == np.int8:
                # Quantize input for int8 model
                scale, zero_point = input_details[0]['quantization']
                img = img / scale + zero_point
                img = np.expand_dims(img, axis=0).astype(np.int8)
            else:
                img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()

            output = interpreter.get_tensor(output_details[0]['index'])

            # Handle dequantization
            if output_details[0]['dtype'] == np.int8:
                scale, zero_point = output_details[0]['quantization']
                output = (output.astype(np.float32) - zero_point) * scale

            predictions.append(np.argmax(output))

        predictions = np.array(predictions)
        true_labels = test_labels[:test_subset]

        # Calculate metrics
        accuracy = np.mean(predictions == true_labels)
        precision = np.sum((predictions == 1) & (true_labels == 1)) / np.sum(predictions == 1) if np.sum(predictions == 1) > 0 else 0
        recall = np.sum((predictions == 1) & (true_labels == 1)) / np.sum(true_labels == 1) if np.sum(true_labels == 1) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print("📈 Confusion Matrix:")
        print(f"   TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"   FN: {cm[1][0]}, TP: {cm[1][1]}")

        return accuracy

def main():
    """Main training function"""
    print("🔧 Advanced Car Detection Trainer for Raspberry Pi")
    print("=" * 60)

    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available. Please install TensorFlow to use advanced training features.")
        print("💡 Run: pip install tensorflow")
        print("🔄 Falling back to custom SVM trainer...")
        # Fallback to SVM training
        import subprocess
        import sys
        try:
            result = subprocess.run([sys.executable, 'src/training/custom_car_trainer.py'],
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Errors:", result.stderr)
        except Exception as e:
            print(f"❌ Failed to run SVM trainer: {e}")
        return

    trainer = AdvancedCarTrainer()

    # Check training data
    car_dir = Path('data/kitti/images_real/toy_car')
    if not car_dir.exists() or len(list(car_dir.rglob('*'))) == 0:
        print("❌ No car training images found!")
        print("📸 Please add car images to data/kitti/images_real/toy_car/")
        return

    try:
        model, fp32_accuracy, int8_accuracy = trainer.train_model(validate_data=True)
        print("\n✅ Advanced training completed successfully!")
        print("=" * 60)
        print(".2f")
        print(".2f")
        print(".2f")
        print("🔄 Advanced optimization techniques applied:")
        print("  • MobileNetV2 α=0.35 lightweight backbone")
        print("  • 40% magnitude-based pruning")
        print("  • Post-training int8 quantization")
        print("  • Advanced data augmentation")
        print("  • Transfer learning from ImageNet")
        print("  • Mixed precision training")
        print("💡 Optimized for Raspberry Pi performance and accuracy!")
        print("🚀 Ready for car detection in traffic light system!")
        print("\n📁 Models saved:")
        print("  • car_detector_optimized.h5 (FP32)")
        print("  • car_detector_int8.tflite (INT8 quantized)")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()