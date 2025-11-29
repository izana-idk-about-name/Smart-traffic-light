#!/usr/bin/env python3
"""
Treinamento OTIMIZADO para detecção de carros de brinquedo
Versão melhorada com múltiplas features e SVM rigoroso
"""

import cv2
import numpy as np
import os
import random
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import time
from typing import List, Tuple

# FIX: Adicionar raiz do projeto ao path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.data_validator import TrainingDataValidator

class OptimizedCarTrainer:
    """
    Trainer otimizado para carros de brinquedo pequenos
    Usa múltiplas features: HOG + Cor + Textura + Forma
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        
        # FIX 1: HOG otimizado para CARROS (horizontal, pequenos)
        self.hog = cv2.HOGDescriptor(
            _winSize=(64, 48),      # Formato horizontal para carros
            _blockSize=(8, 8),      # Blocos menores para objetos pequenos
            _blockStride=(4, 4),    # Stride menor = mais detalhes
            _cellSize=(4, 4),       # Células menores = mais precisão
            _nbins=9
        )
        
        # FIX 2: SVM com parâmetros RIGOROSOS
        self.svm = SVC(
            kernel='rbf',
            C=1000.0,              # Muito rigoroso
            gamma='scale',         # Auto-ajuste
            probability=True,      # Para ter confiança
            class_weight='balanced', # Balanceamento automático
            cache_size=500         # Performance
        )
        
        # FIX 3: Normalização de features (crítico!)
        self.scaler = StandardScaler()
        
        # Performance tracking
        self.feature_extraction_time = []

    def extract_multi_features(self, image_path: str) -> np.ndarray:
        """
        Extração de MÚLTIPLAS features para melhor detecção
        
        Features combinadas:
        1. HOG - Forma e gradientes
        2. Histogramas de cor HSV - Cores dos carros
        3. Textura LBP - Padrões de superfície
        4. Features geométricas - Proporções
        """
        try:
            start_time = time.time()
            
            # Carregar imagem
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            # Converter para diferentes espaços de cor
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # ==== FEATURE 1: HOG (Histograma de Gradientes Orientados) ====
            # Redimensionar para tamanho padrão
            target_size = (96, 64)  # Largura > Altura para carros
            resized_gray = cv2.resize(gray, target_size)
            
            # Aplicar CLAHE para melhorar contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(resized_gray)
            
            # Extrair HOG
            hog_features = self.hog.compute(enhanced)
            if hog_features is None:
                return None
            hog_features = hog_features.flatten()
            
            # ==== FEATURE 2: Histogramas de Cor HSV ====
            resized_hsv = cv2.resize(hsv, target_size)
            
            # Histograma de Hue (matiz) - 16 bins
            hist_h = cv2.calcHist([resized_hsv], [0], None, [16], [0, 180])
            # Histograma de Saturation (saturação) - 16 bins
            hist_s = cv2.calcHist([resized_hsv], [1], None, [16], [0, 256])
            # Histograma de Value (valor) - 16 bins
            hist_v = cv2.calcHist([resized_hsv], [2], None, [16], [0, 256])
            
            # FIX: Normalizar histogramas COM PROTEÇÃO
            color_features = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(),
                hist_v.flatten()
            ])
            
            # CRÍTICO: Proteger divisão por zero
            color_sum = color_features.sum()
            if color_sum > 0:
                color_features = color_features / color_sum
            else:
                color_features = np.zeros_like(color_features)  # Fallback
            
            # ==== FEATURE 3: LBP (Local Binary Patterns) - Textura ====
            def compute_lbp(image, P=8, R=1):
                """Compute LBP features"""
                lbp = np.zeros_like(image)
                for i in range(R, image.shape[0] - R):
                    for j in range(R, image.shape[1] - R):
                        center = image[i, j]
                        code = 0
                        for p in range(P):
                            angle = 2 * np.pi * p / P
                            x = int(i + R * np.cos(angle))
                            y = int(j + R * np.sin(angle))
                            if x >= 0 and x < image.shape[0] and y >= 0 and y < image.shape[1]:
                                if image[x, y] >= center:
                                    code |= (1 << p)
                        lbp[i, j] = code
                return lbp
            
            lbp_image = compute_lbp(resized_gray)
            lbp_hist = cv2.calcHist([lbp_image.astype(np.uint8)], [0], None, [32], [0, 256])
            lbp_features = lbp_hist.flatten()
            # FIX: LBP com proteção
            lbp_sum = lbp_features.sum()
            if lbp_sum > 0:
                lbp_features = lbp_features / lbp_sum
            else:
                lbp_features = np.zeros_like(lbp_features)
            
            # ==== FEATURE 4: Features Geométricas ====
            # Detecção de bordas
            edges = cv2.Canny(resized_gray, 50, 150)
            edge_density = np.sum(edges > 0) / max(edges.shape[0] * edges.shape[1], 1)  # FIX
            
            # Contornos
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                # FIX: Proteção em todas as divisões
                if area > 0:
                    compactness = (perimeter ** 2) / area
                else:
                    compactness = 0
                
                x, y, w, h = cv2.boundingRect(largest_contour)
                if h > 0:
                    aspect_ratio = w / h
                else:
                    aspect_ratio = 1.0
                
                if w > 0 and h > 0:
                    extent = area / (w * h)
                else:
                    extent = 0
            else:
                compactness = 0
                aspect_ratio = 1.0
                extent = 0
            
            geometric_features = np.array([
                edge_density,
                compactness,
                aspect_ratio,
                extent,
                len(contours)  # Número de contornos
            ])
            
            # FIX: Momentos com proteção
            moments = cv2.moments(edges)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform com proteção contra log(0)
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)  # FIX: 1e-10
            
            # ==== COMBINAR TODAS AS FEATURES ====
            combined_features = np.concatenate([
                hog_features,           # ~3000 features
                color_features,         # 48 features
                lbp_features,          # 32 features
                geometric_features,     # 5 features
                hu_moments             # 7 features
            ])
            
            extraction_time = time.time() - start_time
            self.feature_extraction_time.append(extraction_time)
            
            return combined_features.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️  Erro ao processar {image_path}: {e}")
            return None

    def load_balanced_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carregar dados com balanceamento RIGOROSO
        """
        features = []
        labels = []

        # ==== POSITIVOS (Carros) ====
        positive_files = []
        car_dirs = [
            self.data_dir / 'kitti' / 'images_real' / 'toy_car',
            self.data_dir / 'kitti' / 'images_real' / 'toy_f1'
        ]
        
        for car_dir in car_dirs:
            if car_dir.exists():
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    positive_files.extend(list(car_dir.rglob(ext)))

        print(f"📸 Encontrados {len(positive_files)} arquivos de carros")

        # Limitar positivos (evitar overfitting)
        max_positives = 400
        if len(positive_files) > max_positives:
            positive_files = random.sample(positive_files, max_positives)
            print(f"   Limitando a {max_positives} carros")

        positive_count = 0
        for i, img_path in enumerate(positive_files):
            feat = self.extract_multi_features(str(img_path))
            if feat is not None:
                features.append(feat)
                labels.append(1)  # Carro
                positive_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"   Carros: {i + 1}/{len(positive_files)} ({(i+1)/len(positive_files)*100:.1f}%)")

        print(f"✅ Carregados {positive_count} carros válidos")

        # ==== NEGATIVOS (Sem carros) ====
        negative_files = []
        negative_dirs = [
            self.data_dir / 'negativo',
            self.data_dir / 'negativo_camera',
            self.data_dir / 'negativo_synthetic',
            self.data_dir / 'imagens_originais'
        ]

        for neg_dir in negative_dirs:
            if neg_dir.exists():
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    negative_files.extend(list(neg_dir.glob(ext)))

        print(f"📸 Encontrados {len(negative_files)} arquivos negativos")

        # Balanceamento 2:1 (mais negativos para robustez)
        target_negatives = positive_count * 2
        
        if len(negative_files) < positive_count:
            print(f"\n❌ ERRO: Apenas {len(negative_files)} negativos vs {positive_count} positivos")
            print(f"   MÍNIMO necessário: {positive_count} negativos")
            print(f"   RECOMENDADO: {target_negatives} negativos")
            print(f"\n💡 Execute: python3 scripts/capture_negative_samples.py")
            raise ValueError("Dados insuficientes - capture mais negativos!")
        
        if len(negative_files) > target_negatives:
            negative_files = random.sample(negative_files, target_negatives)

        negative_count = 0
        for i, img_path in enumerate(negative_files):
            feat = self.extract_multi_features(str(img_path))
            if feat is not None:
                features.append(feat)
                labels.append(0)  # Sem carro
                negative_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"   Negativos: {i + 1}/{len(negative_files)} ({(i+1)/len(negative_files)*100:.1f}%)")

        print(f"✅ Carregados {negative_count} negativos válidos")
        
        # Validação de balanceamento
        ratio = negative_count / positive_count
        print(f"\n📊 BALANCEAMENTO:")
        print(f"   Positivos: {positive_count}")
        print(f"   Negativos: {negative_count}")
        print(f"   Razão: {ratio:.2f}:1")
        
        if ratio < 1.0:
            raise ValueError("❌ Dataset desbalanceado! Capture mais negativos.")
        elif ratio < 1.5:
            print("⚠️  Balanceamento mínimo - recomendado 2:1")
        else:
            print("✅ Dataset bem balanceado!")

        return np.array(features), np.array(labels)

    def optimize_hyperparameters(self, X_train, y_train):
        """
        Otimização automática de hiperparâmetros com GridSearch
        """
        print("\n🔧 Otimizando hiperparâmetros do SVM...")
        
        param_grid = {
            'C': [100, 500, 1000, 2000],
            'gamma': ['scale', 'auto', 0.001, 0.0001],
            'kernel': ['rbf']
        }
        
        grid_search = GridSearchCV(
            SVC(probability=True, class_weight='balanced'),
            param_grid,
            cv=3,  # 3-fold cross-validation
            scoring='f1',
            n_jobs=-1,  # Usar todos os cores
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Melhores parâmetros encontrados:")
        print(f"   C: {grid_search.best_params_['C']}")
        print(f"   Gamma: {grid_search.best_params_['gamma']}")
        print(f"   Score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_

    def train_model(self, test_size: float = 0.25, optimize_params: bool = True):
        """
        Treinar modelo otimizado
        """
        print("🚗 Iniciando treinamento OTIMIZADO para carros de brinquedo")
        print("=" * 60)

        # Carregar dados balanceados
        features, labels = self.load_balanced_data()
        
        print(f"\n📊 Dataset final: {len(features)} amostras")
        unique, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique, counts):
            name = "Carros" if cls == 1 else "Negativos"
            print(f"   {name}: {count} ({count/len(labels)*100:.1f}%)")

        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=test_size,
            random_state=42,
            stratify=labels  # Manter proporção
        )

        print(f"\n📊 Split:")
        print(f"   Treino: {len(X_train)} ({len(X_train)/len(features)*100:.0f}%)")
        print(f"   Teste: {len(X_test)} ({len(X_test)/len(features)*100:.0f}%)")

        # CRÍTICO: Normalizar features
        print("\n🔧 Normalizando features (StandardScaler)...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Otimizar hiperparâmetros
        if optimize_params:
            self.svm = self.optimize_hyperparameters(X_train_scaled, y_train)
        else:
            # Treinar com parâmetros fixos
            print("\n🚗 Treinando SVM...")
            start_time = time.time()
            self.svm.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            print(f"✅ Treinamento concluído em {training_time:.2f}s")

        # Avaliar
        print("\n📊 AVALIAÇÃO:")
        
        # Treino
        train_pred = self.svm.predict(X_train_scaled)
        train_acc = np.mean(train_pred == y_train)
        print(f"   Acurácia Treino: {train_acc*100:.1f}%")
        
        # Teste
        test_pred = self.svm.predict(X_test_scaled)
        test_acc = np.mean(test_pred == y_test)
        print(f"   Acurácia Teste: {test_acc*100:.1f}%")
        
        # Verificar overfitting
        if train_acc - test_acc > 0.1:
            print(f"\n⚠️  ALERTA: Possível overfitting ({(train_acc-test_acc)*100:.1f}% diferença)")
        
        # Relatório detalhado
        print(f"\n📋 Relatório de Classificação:")
        print(classification_report(y_test, test_pred,
                                   target_names=['Sem Carro', 'Com Carro'],
                                   digits=3))

        # Matriz de confusão
        cm = confusion_matrix(y_test, test_pred)
        print(f"\n🎯 Matriz de Confusão:")
        print(f"                 Previsto")
        print(f"              Neg     Pos")
        print(f"Real   Neg   {cm[0,0]:4d}    {cm[0,1]:4d}   (Verdadeiro Neg | Falso Pos)")
        print(f"       Pos   {cm[1,0]:4d}    {cm[1,1]:4d}   (Falso Neg | Verdadeiro Pos)")
        
        # Análise de erros
        false_positives = cm[0, 1]
        false_negatives = cm[1, 0]
        print(f"\n📉 Análise de Erros:")
        print(f"   Falsos Positivos: {false_positives} ({false_positives/cm[0].sum()*100:.1f}% dos negativos)")
        print(f"   Falsos Negativos: {false_negatives} ({false_negatives/cm[1].sum()*100:.1f}% dos positivos)")

        # Salvar modelo
        self.save_model()
        
        # Performance
        if self.feature_extraction_time:
            avg_feat_time = np.mean(self.feature_extraction_time)
            print(f"\n⏱️  Performance:")
            print(f"   Extração de features: {avg_feat_time*1000:.1f}ms por imagem")

        return test_acc

    def save_model(self, filename='src/models/custom_car_detector_optimized.pkl'):
        """Salvar modelo completo"""
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'hog_params': {
                'winSize': (64, 48),
                'blockSize': (8, 8),
                'blockStride': (4, 4),
                'cellSize': (4, 4),
                'nbins': 9
            }
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n💾 Modelo salvo em {filename}")
        print(f"   Tamanho: {Path(filename).stat().st_size / 1024:.1f} KB")

    def load_model(self, filename='src/models/custom_car_detector_optimized.pkl'):
        """Carregar modelo completo"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.svm = model_data['svm']
            self.scaler = model_data['scaler']
            
            # Recriar HOG
            params = model_data['hog_params']
            self.hog = cv2.HOGDescriptor(
                _winSize=params['winSize'],
                _blockSize=params['blockSize'],
                _blockStride=params['blockStride'],
                _cellSize=params['cellSize'],
                _nbins=params['nbins']
            )
            
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar: {e}")
            return False

    def predict(self, image_path: str, threshold: float = 0.75) -> Tuple[int, float]:
        """
        Predição com threshold RIGOROSO
        
        Args:
            image_path: Caminho da imagem
            threshold: Limiar de decisão (padrão 0.75 - rigoroso)
        
        Returns:
            (predição, confiança)
        """
        try:
            # Extrair features
            features = self.extract_multi_features(image_path)
            if features is None:
                return 0, 0.0

            # Normalizar
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predição com probabilidade
            probabilities = self.svm.predict_proba(features_scaled)[0]
            car_probability = probabilities[1]
            
            # Decisão com threshold rigoroso
            prediction = int(car_probability > threshold)
            
            return prediction, car_probability

        except Exception as e:
            print(f"⚠️  Erro: {e}")
            return 0, 0.0


def main():
    """Função principal"""
    print("🎯 Optimized Car Trainer - Versão Melhorada")
    print("=" * 60)

    trainer = OptimizedCarTrainer()

    # Verificar dados
    car_dir = Path('data/kitti/images_real/toy_car')
    if not car_dir.exists():
        print("❌ Diretório de carros não encontrado!")
        return

    neg_dir = Path('data/negativo')
    if not neg_dir.exists():
        print("❌ Capture amostras negativas primeiro!")
        print("   Execute: python3 scripts/capture_negative_samples.py")
        return

    try:
        # FIX: Treinar SEM otimização de hiperparâmetros (muito mais rápido)
        print("\n⚡ Modo rápido: usando parâmetros pré-otimizados")
        accuracy = trainer.train_model(optimize_params=False)  # ← MUDANÇA AQUI
        
        if accuracy > 0.90:
            print(f"\n🎉 EXCELENTE! Precisão: {accuracy*100:.1f}%")
        elif accuracy > 0.85:
            print(f"\n✅ BOM! Precisão: {accuracy*100:.1f}%")
        else:
            print(f"\n⚠️  Precisão baixa: {accuracy*100:.1f}%")
            print("   Capture mais dados e retreine")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()