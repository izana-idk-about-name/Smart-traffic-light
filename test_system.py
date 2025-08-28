#!/usr/bin/env python3
"""
Script de teste completo do sistema de controle de semáforos
Verifica todos os componentes antes da instalação no Raspberry Pi
"""

import os
import sys
import cv2
import time
import logging
from pathlib import Path

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemTester:
    def __init__(self):
        self.tests_passed = 0
        self.tests_total = 0
        
    def test(self, test_name):
        """Decorator para executar testes"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                self.tests_total += 1
                try:
                    logger.info(f"🧪 Executando teste: {test_name}")
                    result = func(*args, **kwargs)
                    if result is not False:
                        logger.info(f"✅ {test_name} - PASSOU")
                        self.tests_passed += 1
                        return True
                    else:
                        logger.error(f"❌ {test_name} - FALHOU")
                        return False
                except Exception as e:
                    logger.error(f"❌ {test_name} - ERRO: {e}")
                    return False
            return wrapper
        return decorator
    
    def run_all_tests(self):
        """Executar todos os testes"""
        logger.info("=" * 50)
        logger.info("INICIANDO TESTES DO SISTEMA")
        logger.info("=" * 50)
        
        # Executar testes
        self.test_dependencies()
        self.test_camera_access()
        self.test_model_loading()
        self.test_configuration()
        self.test_file_structure()
        
        # Resultados
        logger.info("=" * 50)
        logger.info("RESULTADOS DOS TESTES")
        logger.info("=" * 50)
        logger.info(f"Testes passados: {self.tests_passed}/{self.tests_total}")
        
        if self.tests_passed == self.tests_total:
            logger.info("🎉 Todos os testes passaram! Sistema pronto para uso.")
            return True
        else:
            logger.warning("⚠️  Alguns testes falharam. Verifique os logs acima.")
            return False
    
    @test("Dependências Python")
    def test_dependencies(self):
        """Verificar se todas as dependências estão instaladas"""
        required_packages = [
            'cv2', 'numpy', 'PIL', 'sklearn', 'joblib',
            'scipy', 'matplotlib', 'requests', 'websocket',
            'dotenv', 'psutil', 'colorlog'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.error(f"Pacotes faltando: {', '.join(missing)}")
            return False
        
        return True
    
    @test("Acesso às Câmeras")
    def test_camera_access(self):
        """Verificar se as câmeras estão acessíveis"""
        cameras_found = 0
        
        for i in range(4):  # Testar até 4 câmeras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                    logger.info(f"  📷 Câmera {i}: {width}x{height} - OK")
                    cameras_found += 1
                else:
                    logger.warning(f"  ⚠️  Câmera {i}: aberta mas não captura imagem")
                cap.release()
            else:
                logger.debug(f"  ❌ Câmera {i}: não disponível")
        
        if cameras_found >= 2:
            logger.info(f"  ✅ {cameras_found} câmeras encontradas")
            return True
        else:
            logger.error(f"  ❌ Apenas {cameras_found} câmeras encontradas (mínimo 2)")
            return False
    
    @test("Carregamento do Modelo")
    def test_model_loading(self):
        """Verificar se o modelo de identificação carrega corretamente"""
        try:
            from src.models.car_identify import create_car_identifier
            
            # Testar modelo para Raspberry Pi
            identifier_rpi = create_car_identifier('rpi')
            logger.info("  ✅ Modelo para Raspberry Pi carregado")
            
            # Testar modelo para desktop
            identifier_desktop = create_car_identifier('desktop')
            logger.info("  ✅ Modelo para desktop carregado")
            
            return True
        except Exception as e:
            logger.error(f"  ❌ Erro ao carregar modelo: {e}")
            return False
    
    @test("Configuração do Sistema")
    def test_configuration(self):
        """Verificar configurações do sistema"""
        try:
            from src.settings.rpi_config import (
                IS_RASPBERRY_PI, CAMERA_SETTINGS, PROCESSING_SETTINGS,
                MODEL_SETTINGS, NETWORK_SETTINGS
            )
            
            logger.info(f"  🖥️  Raspberry Pi detectado: {IS_RASPBERRY_PI}")
            logger.info(f"  📹 Configuração câmera: {CAMERA_SETTINGS['width']}x{CAMERA_SETTINGS['height']}@{CAMERA_SETTINGS['fps']}fps")
            logger.info(f"  ⚙️  Intervalo decisão: {PROCESSING_SETTINGS['decision_interval']}s")
            
            return True
        except Exception as e:
            logger.error(f"  ❌ Erro na configuração: {e}")
            return False
    
    @test("Estrutura de Arquivos")
    def test_file_structure(self):
        """Verificar se todos os arquivos necessários existem"""
        required_files = [
            'main.py',
            'src/models/car_identify.py',
            'src/application/camera.py',
            'src/application/comunicator.py',
            'src/settings/config.py',
            'src/settings/rpi_config.py',
            'requirements.txt',
            '.env'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"  ❌ Arquivos faltando: {', '.join(missing_files)}")
            return False
        
        logger.info("  ✅ Todos os arquivos necessários encontrados")
        return True

def main():
    """Função principal de teste"""
    tester = SystemTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 SISTEMA PRONTO PARA USO!")
        print("=" * 50)
        print("\nPróximos passos:")
        print("1. Execute: python3 main.py")
        print("2. Ou use: ./start.sh")
        print("3. Para teste: MODO=development python3 main.py")
    else:
        print("\n" + "=" * 50)
        print("⚠️  CORRIJA OS PROBLEMAS ANTES DE CONTINUAR")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()