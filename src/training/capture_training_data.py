#!/usr/bin/env python3
"""
Script para captura de dados de treinamento
Captura imagens das câmeras para treinamento do modelo
"""

import cv2
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

class TrainingDataCapture:
    def __init__(self, camera_index=0, output_dir='src/Data/images'):
        self.camera_index = camera_index
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar subdiretórios
        for category in ['cars', 'trucks', 'motorcycles', 'background']:
            (self.output_dir / category).mkdir(exist_ok=True)
        
        self.cap = None
        
    def initialize_camera(self):
        """Inicializar câmera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir a câmera {self.camera_index}")
        
        # Configurar resolução
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return True
    
    def capture_image(self, category, filename=None):
        """Capturar uma imagem"""
        if not self.cap:
            self.initialize_camera()
        
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Não foi possível capturar imagem")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_cam{self.camera_index}.jpg"
        
        output_path = self.output_dir / category / filename
        cv2.imwrite(str(output_path), frame)
        
        return str(output_path)
    
    def interactive_capture(self):
        """Captura interativa com interface"""
        if not self.cap:
            self.initialize_camera()
        
        categories = {
            ord('1'): 'cars',
            ord('2'): 'trucks', 
            ord('3'): 'motorcycles',
            ord('0'): 'background'
        }
        
        print("=== Captura de Dados de Treinamento ===")
        print("Pressione:")
        print("  1 - Carro")
        print("  2 - Caminhão")
        print("  3 - Motocicleta")
        print("  0 - Sem veículo (background)")
        print("  ESC - Sair")
        print("")
        
        count = {cat: 0 for cat in categories.values()}
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Erro ao capturar frame")
                break
            
            # Mostrar preview
            display_frame = frame.copy()
            cv2.putText(display_frame, "Pressione 1,2,3,0 ou ESC", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar contador
            y_offset = 60
            for cat, cnt in count.items():
                cv2.putText(display_frame, f"{cat}: {cnt}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            cv2.imshow('Training Data Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key in categories:
                category = categories[key]
                try:
                    filepath = self.capture_image(category)
                    count[category] += 1
                    print(f"✓ Capturado: {category} - {os.path.basename(filepath)}")
                except Exception as e:
                    print(f"Erro ao capturar: {e}")
        
        self.cleanup()
        
        print("\n=== Resumo da Captura ===")
        for cat, cnt in count.items():
            print(f"{cat}: {cnt} imagens")
    
    def batch_capture(self, category, count=100, delay=1):
        """Captura em lote"""
        if not self.cap:
            self.initialize_camera()
        
        print(f"Capturando {count} imagens para categoria '{category}'...")
        
        for i in range(count):
            try:
                filepath = self.capture_image(category, f"{category}_{i+1:04d}.jpg")
                print(f"  {i+1}/{count}: {os.path.basename(filepath)}")
                time.sleep(delay)
            except Exception as e:
                print(f"Erro na captura {i+1}: {e}")
                break
        
        self.cleanup()
        print("Captura em lote concluída!")
    
    def cleanup(self):
        """Limpar recursos"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()

def train_car_detector():
    """Train a custom car detection model using collected data"""
    print("🚗 Starting car detector training...")

    # Check if training data exists
    images_dir = Path('src/Data/images')
    if not images_dir.exists():
        print("❌ Training data not found. Run data capture first.")
        return

    car_images = list(images_dir.glob('cars/*.jpg'))
    background_images = list(images_dir.glob('background/*.jpg'))

    print(f"📊 Found {len(car_images)} car images and {len(background_images)} background images")

    if len(car_images) < 100 or len(background_images) < 100:
        print("⚠️  Need at least 100 images per category for training")
        print("📸 Collect more data first using: python -m src.training.capture_training_data")
        return

    # For now, we'll use the pre-trained models
    # In a real implementation, you would train a custom model here
    print("✅ Using pre-trained ML models (sufficient for most use cases)")
    print("💡 Custom training can be implemented if needed for specialized scenarios")

def main():
    parser = argparse.ArgumentParser(description='Captura dados de treinamento')
    parser.add_argument('--camera', type=int, default=0, help='Índice da câmera')
    parser.add_argument('--output', default='src/Data/images', help='Diretório de saída')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                       help='Modo de captura')
    parser.add_argument('--category', choices=['cars', 'trucks', 'motorcycles', 'background'],
                       help='Categoria para captura em lote')
    parser.add_argument('--count', type=int, default=100, help='Número de imagens para captura em lote')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay entre capturas (segundos)')
    parser.add_argument('--train', action='store_true', help='Treinar modelo após captura de dados')
    
    args = parser.parse_args()
    
    capture = TrainingDataCapture(args.camera, args.output)
    
    if args.mode == 'interactive':
        capture.interactive_capture()
    else:
        if not args.category:
            print("Erro: --category é obrigatório no modo batch")
            return

        capture.batch_capture(args.category, args.count, args.delay)

    # Train model if requested
    if args.train:
        print("\n🚀 Training phase...")
        train_car_detector()

if __name__ == "__main__":
    main()