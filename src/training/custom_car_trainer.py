#!/usr/bin/env python3
"""
Treinamento personalizado leve para detecção de carros
Usa OpenCV ML (SVM) com features HOG - otimizado para Raspberry Pi
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import time
from typing import List, Tuple
from src.training.data_validator import TrainingDataValidator, validate_dataset_quick

class LightweightCarTrainer:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 128),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9
        )

        # Initialize SVM with optimized parameters for RPi
        self.svm = cv2.ml.SVM_create()
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setKernel(cv2.ml.SVM_RBF)
        self.svm.setC(10.0)  # Aumenta a penalidade para erros
        self.svm.setGamma(0.01)  # Menor gamma para maior generalização
        self.svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-6))

    def extract_hog_features(self, image_path: str) -> np.ndarray:
        """Extract HOG features from image with robust preprocessing for real KITTI data"""
        try:
            image = cv2.imread(str(image_path))
            if image is None or image.size == 0:
                return None

            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Ensure minimum size - resize if too small
            if gray.shape[0] < 64 or gray.shape[1] < 64:
                # Resize to minimum 64x64
                gray = cv2.resize(gray, (max(64, gray.shape[1]), max(64, gray.shape[0])), interpolation=cv2.INTER_LINEAR)

            # For real KITTI images, apply different preprocessing
            if 'kitti' in str(image_path).lower() or str(image_path).endswith('.png'):
                # KITTI images are already good quality, less aggressive preprocessing
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                enhanced = blurred
            else:
                # For synthetic/toy images, use CLAHE enhancement
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(blurred)

            # Resize to standard size (keeping aspect ratio)
            h, w = enhanced.shape
            if h > w:
                new_h = 128
                new_w = int(w * (128 / h))
            else:
                new_w = 128
                new_h = int(h * (128 / w))

            # Ensure minimum dimensions
            new_w = max(new_w, 64)
            new_h = max(new_h, 64)

            resized = cv2.resize(enhanced, (new_w, new_h))

            # Pad to 128x128 if necessary
            if resized.shape[0] < 128 or resized.shape[1] < 128:
                padded = np.zeros((128, 128), dtype=np.uint8)
                h_pad = (128 - resized.shape[0]) // 2
                w_pad = (128 - resized.shape[1]) // 2
                padded[h_pad:h_pad+resized.shape[0], w_pad:w_pad+resized.shape[1]] = resized
                final_image = padded
            else:
                final_image = resized

            # Extract HOG features
            features = self.hog.compute(final_image)

            if features is None or len(features) == 0:
                return None

            return features.flatten()

        except Exception as e:
            # Return None on any error instead of crashing
            return None

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare training data with proper positive/negative balance"""
        features = []
        labels = []

        # Load car images (positive samples)
        car_files = []
        carro_dir = self.data_dir / 'kitti' / 'images_real' / 'toy_car'
        if carro_dir.exists():
            # Recursively find all image files in the carro directory and subdirectories
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.avif']:
                car_files.extend(list(carro_dir.rglob(ext)))

        # Load toy_f1 images (positive samples)
        toy_f1_dir = self.data_dir / 'kitti' / 'images_real' / 'toy_f1'
        if toy_f1_dir.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.avif']:
                car_files.extend(list(toy_f1_dir.rglob(ext)))

        if car_files:
            # For KITTI data, use more samples since they're high quality
            if any('kitti' in str(f).lower() or str(f).endswith('.png') for f in car_files[:10]):
                max_images = len(car_files)  # Use all KITTI images
                print("🎯 KITTI dataset detectado - usando todas as imagens disponíveis!")
            else:
                # For synthetic/toy data, limit to 1/4
                max_images = len(car_files) // 4

            car_files_subset = random.sample(car_files, min(max_images, len(car_files)))
            print(f"📸 Loading {len(car_files_subset)} car images (POSITIVE samples)...")
            valid_car_count = 0
            for i, img_path in enumerate(car_files_subset):
                feat = self.extract_hog_features(str(img_path))
                if feat is not None:
                    features.append(feat)
                    labels.append(1)  # Car = 1
                    valid_car_count += 1
                else:
                    print(f"⚠️  Skipping invalid car image: {img_path}")

                # Print progress every 500 images
                if (i + 1) % 500 == 0:
                    progress = (i + 1) / len(car_files_subset) * 100
                    print(f"   Progress: {progress:.1f}%")
            print(f"✅ Valid car images loaded: {valid_car_count}")

        # Load NEGATIVE samples (images WITHOUT cars) from data/negativo/
        negativo_dir = self.data_dir / 'negativo'
        negative_files = []
        
        if negativo_dir.exists():
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.avif']:
                negative_files.extend(list(negativo_dir.glob(ext)))
            
            print(f"📸 Loading {len(negative_files)} NEGATIVE samples from data/negativo/...")
            valid_negative_count = 0
            for i, img_path in enumerate(negative_files):
                feat = self.extract_hog_features(str(img_path))
                if feat is not None:
                    features.append(feat)
                    labels.append(0)  # No car = 0
                    valid_negative_count += 1
                else:
                    print(f"⚠️  Skipping invalid negative image: {img_path}")
                
                # Print progress every 10 images
                if (i + 1) % 10 == 0:
                    progress = (i + 1) / len(negative_files) * 100
                    print(f"   Progress: {progress:.1f}%")
            print(f"✅ Valid negative images loaded: {valid_negative_count}")
        
        # Load NEGATIVE samples from camera captures (data/negativo_camera/)
        negativo_camera_dir = self.data_dir / 'negativo_camera'
        if negativo_camera_dir.exists():
            camera_negative_files = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.avif']:
                camera_negative_files.extend(list(negativo_camera_dir.glob(ext)))
            
            if camera_negative_files:
                print(f"📸 Loading {len(camera_negative_files)} NEGATIVE samples from camera...")
                for i, img_path in enumerate(camera_negative_files):
                    feat = self.extract_hog_features(str(img_path))
                    if feat is not None:
                        features.append(feat)
                        labels.append(0)  # No car = 0
                        valid_negative_count += 1
                    
                    # Print progress every 10 images
                    if (i + 1) % 10 == 0:
                        progress = (i + 1) / len(camera_negative_files) * 100
                        print(f"   Progress: {progress:.1f}%")
                print(f"✅ Camera negative images loaded: {len(camera_negative_files)}")

        # Also load background images from imagens_originais (if they exist)
        bg_dir = self.data_dir / 'imagens_originais'
        if bg_dir.exists():
            bg_files = []
            for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp', '*.avif']:
                bg_files.extend(list(bg_dir.glob(ext)))

            if bg_files:
                print(f"📸 Loading {len(bg_files)} additional background images...")
                for i, img_path in enumerate(bg_files):
                    feat = self.extract_hog_features(str(img_path))
                    if feat is not None:
                        features.append(feat)
                        labels.append(0)  # Background = 0

                    # Print progress every 100 images
                    if (i + 1) % 100 == 0:
                        progress = (i + 1) / len(bg_files) * 100
                        print(f"   Progress: {progress:.1f}%")

        if len(features) == 0:
            raise ValueError("❌ No training images found!")

        # Show class distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"\n📊 Class Distribution:")
        for cls, count in zip(unique, counts):
            class_name = "CARS (positive)" if cls == 1 else "NO CARS (negative)"
            print(f"   {class_name}: {count} samples")
        
        return np.array(features), np.array(labels)

    def augment_data(self, features: np.ndarray, labels: np.ndarray,
                    augmentation_factor: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Data augmentation: noise, flip, scale"""
        augmented_features = []
        augmented_labels = []

        for feat, label in zip(features, labels):
            # Original sample
            augmented_features.append(feat)
            augmented_labels.append(label)

            # Add noise variations
            for _ in range(augmentation_factor - 1):
                noise = np.random.normal(0, 0.08, feat.shape)
                noisy_feat = feat + noise
                augmented_features.append(noisy_feat)
                augmented_labels.append(label)

            # Flip (simulado via reversão dos vetores HOG)
            flipped_feat = feat[::-1]
            augmented_features.append(flipped_feat)
            augmented_labels.append(label)

            # Scale (simulado via multiplicação)
            scaled_feat = feat * np.random.uniform(0.9, 1.1, feat.shape)
            augmented_features.append(scaled_feat)
            augmented_labels.append(label)

        return np.array(augmented_features), np.array(augmented_labels)

    def train_model(self, test_size: float = 0.3, validate_data: bool = True):
        """Train the SVM model with optional data validation"""
        print("🚗 Starting lightweight car detection training...")

        # Validate training data if requested
        if validate_data:
            print("\n🔍 Validating training data before training...")
            try:
                validator = TrainingDataValidator(
                    min_samples_per_class=50,  # Reasonable minimum for SVM
                    min_image_width=32,
                    min_image_height=32,
                    max_class_imbalance=10.0,
                    check_duplicates=True
                )
                
                # Validate car data
                car_dir = self.data_dir / 'kitti' / 'images_real'
                if car_dir.exists():
                    result = validator.validate_dataset(str(car_dir), class_dirs=['toy_car', 'toy_f1'])
                    
                    print(f"\n{'='*60}")
                    print("Validação dos Dados de Treinamento")
                    print(f"{'='*60}")
                    print(f"Total: {result.total_samples} | Válidos: {result.valid_samples}")
                    print(f"Status: {'✅ VÁLIDO' if result.is_valid else '⚠️  COM AVISOS'}")
                    
                    if result.warnings:
                        print("\n⚠️  Avisos:")
                        for warning in result.warnings[:3]:
                            print(f"  - {warning}")
                    
                    if result.errors:
                        print("\n❌ Erros Críticos:")
                        for error in result.errors:
                            print(f"  - {error}")
                        print("\n🛑 Treinamento cancelado devido a erros nos dados")
                        print("💡 Corrija os erros e tente novamente")
                        return None, None
                    
                    if result.valid_samples < 50:
                        print(f"\n⚠️  Apenas {result.valid_samples} amostras válidas encontradas")
                        print("   Recomendado: pelo menos 100 amostras para treinamento robusto")
                        print("   Continuando, mas resultados podem ser limitados...")
                    
                    print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"⚠️  Erro na validação: {e}")
                print("   Continuando com treinamento (sem validação)...")

        # Load data
        features, labels = self.load_training_data()
        print(f"📊 Dataset: {len(features)} samples, {np.sum(labels==1)} cars, {np.sum(labels==0)} background")

        # Data augmentation
        features, labels = self.augment_data(features, labels, augmentation_factor=3)
        print(f"📊 After augmentation: {len(features)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )

        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")

        # Convert data types for OpenCV SVM
        X_train = X_train.astype(np.float32)
        y_train = y_train.astype(np.int32)

        # Train SVM
        print("🚗 Training SVM model...")
        start_time = time.time()
        self.svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
        training_time = time.time() - start_time
        print("✅ SVM training completed!")

        print(".2f")
        # Convert test data type
        X_test = X_test.astype(np.float32)

        # Evaluate
        _, predictions = self.svm.predict(X_test)
        predictions = predictions.flatten().astype(int)

        # Calculate accuracy
        accuracy = np.mean(predictions == y_test)
        print(".2f")
        # Save model
        self.save_model()

        return accuracy, training_time

    def save_model(self, filename='src/models/custom_car_detector.yml'):
        """Save trained SVM model"""
        self.svm.save(str(filename))
        print(f"💾 Model saved to {filename}")

    def load_model(self, filename='src/models/custom_car_detector.yml'):
        """Load trained SVM model"""
        if os.path.exists(filename):
            self.svm = cv2.ml.SVM_load(filename)
            return True
        return False

    def predict(self, image_path: str) -> Tuple[int, float]:
        """Predict if image contains a car with better error handling"""
        try:
            features = self.extract_hog_features(image_path)
            if features is None or len(features) == 0:
                return -1, 0.0  # Error indicator

            _, result = self.svm.predict(features.reshape(1, -1))
            confidence = float(result[0][0])

            # Use lower threshold for recall, but return confidence for post-processing
            return int(confidence > 0.5), confidence

        except Exception as e:
            # Return error indicator instead of crashing
            return -1, 0.0

def create_synthetic_background_samples(num_samples: int = 50):
    """Create synthetic background samples for better training"""
    print(f"🎨 Creating {num_samples} synthetic background samples...")

    bg_dir = Path('data/imagens_originais')
    bg_dir.mkdir(exist_ok=True)

    for i in range(num_samples):
        # Create random noise image
        noise = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        filename = f"synthetic_bg_{i:04d}.jpg"
        cv2.imwrite(str(bg_dir / filename), noise)

    print("✅ Synthetic background samples created!")

def main():
    """Main training function"""
    print("🔧 Lightweight Car Detection Trainer for Raspberry Pi")
    print("=" * 50)

    trainer = LightweightCarTrainer()

    # Check if training data exists
    carro_dir = Path('data/kitti/images_real/toy_car')
    if not carro_dir.exists() or len(list(carro_dir.rglob('*'))) == 0:
        print("❌ No car training images found!")
        print("📸 Please add car images to data/kitti/images_real/toy_car/")
        return

    # Create background samples if needed
    bg_dir = Path('data/imagens_originais')
    if not bg_dir.exists() or len(list(bg_dir.glob('*'))) < 10:
        print("⚠️  Limited background samples. Creating synthetic ones...")
        create_synthetic_background_samples()

    # Train model with validation
    try:
        result = trainer.train_model(validate_data=True)
        if result is None or result[0] is None:
            print("\n❌ Training aborted due to data validation errors")
            print("💡 Fix the data issues and try again")
            return
        
        accuracy, training_time = result
        print("\n✅ Training completed successfully!")
        print(f"📊 Final Accuracy: {accuracy*100:.2f}%")
        print(f"⏱️  Training Time: {training_time:.2f}s")
        print("💡 This model is optimized for Raspberry Pi performance!")
        print("🔄 You can now use it in your car detection system!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

def demo_camera_detection():
    """Demonstração ao vivo da detecção de carros usando a IA customizada"""
    import cv2
    trainer = LightweightCarTrainer()
    if not trainer.load_model():
        print("❌ Modelo não encontrado. Treine antes de rodar a demo.")
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Não foi possível abrir a câmera.")
        return
    print("🔴 Pressione ESC para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Falha ao capturar frame da câmera.")
            break
        # Salva frame temporário para predição
        temp_path = "/tmp/frame_demo.jpg"
        cv2.imwrite(temp_path, frame)
        pred, conf = trainer.predict(temp_path)
        label = f"Carro: {'Sim' if pred == 1 else 'Não'} ({conf:.2f})"
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.imshow("Detecção de Carro (IA Customizada)", frame)
        # Se não houver suporte a GUI, salve o frame com resultado
        out_path = "/tmp/frame_demo_result.jpg"
        cv2.imwrite(out_path, frame)
        print(f"Frame salvo em {out_path} | {label}")
        # Sem waitKey, apenas salva frames continuamente
    cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_camera_detection()
    else:
        main()

if __name__ == "__main__":
    exit(main())