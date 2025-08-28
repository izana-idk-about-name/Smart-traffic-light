import cv2
import os
import time
import threading
import signal
import sys
from src.models.car_identify import create_car_identifier
from src.application.comunicator import OrchestratorComunicator
from src.settings.rpi_config import CAMERA_SETTINGS, PROCESSING_SETTINGS, MODEL_SETTINGS, NETWORK_SETTINGS, IS_RASPBERRY_PI

class TrafficLightController:
    def __init__(self, camera_a_index=0, camera_b_index=1, orchestrator_host='localhost', orchestrator_port=9000):
        self.camera_a_index = camera_a_index
        self.camera_b_index = camera_b_index
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port
        
        # Initialize cameras with optimized settings
        self.camera_a = None
        self.camera_b = None
        
        # Initialize car identifiers with platform-specific optimizations
        self.car_identifier_a = create_car_identifier('rpi' if IS_RASPBERRY_PI else 'desktop')
        self.car_identifier_b = create_car_identifier('rpi' if IS_RASPBERRY_PI else 'desktop')
        
        # Initialize orchestrator communicator
        self.communicator = OrchestratorComunicator(
            host=orchestrator_host,
            port=orchestrator_port,
            use_websocket=NETWORK_SETTINGS['use_websocket']
        )
        
        # Control flags
        self.running = False
        self.current_light = None
        
        # Performance tracking
        self.cycle_count = 0
        self.start_time = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nRecebido sinal {signum}. Encerrando...")
        self.running = False
    
    def initialize_cameras(self):
        """Initialize cameras with optimized settings"""
        try:
            # Camera A
            self.camera_a = cv2.VideoCapture(self.camera_a_index)
            if IS_RASPBERRY_PI:
                # Raspberry Pi camera optimizations
                self.camera_a.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
                self.camera_a.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])
                self.camera_a.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
                self.camera_a.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_SETTINGS['buffer_size'])
            else:
                # Desktop settings
                self.camera_a.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_a.set(cv2.CAP_PROP_FPS, 15)
            
            # Camera B
            self.camera_b = cv2.VideoCapture(self.camera_b_index)
            if IS_RASPBERRY_PI:
                self.camera_b.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
                self.camera_b.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])
                self.camera_b.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
                self.camera_b.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_SETTINGS['buffer_size'])
            else:
                self.camera_b.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_b.set(cv2.CAP_PROP_FPS, 15)
            
            # Test cameras
            ret_a, _ = self.camera_a.read()
            ret_b, _ = self.camera_b.read()
            
            if not ret_a or not ret_b:
                raise Exception("Failed to read from one or both cameras")
                
            print("Câmeras inicializadas com sucesso")
            return True
            
        except Exception as e:
            print(f"Erro ao inicializar câmeras: {e}")
            return False
    
    def process_frame(self, camera, car_identifier, camera_name):
        """Process a single frame from a camera"""
        ret, frame = camera.read()
        if not ret:
            print(f"Erro ao capturar frame da câmera {camera_name}")
            return 0
        
        # Count cars
        car_count = car_identifier.count_cars(frame)
        
        # Log processing time occasionally
        if car_identifier.frame_count % 30 == 0:
            avg_time = car_identifier.get_average_processing_time()
            print(f"Câmera {camera_name}: {car_count} carros detectados (tempo médio: {avg_time:.3f}s)")
        
        return car_count
    
    def make_decision(self, count_a, count_b):
        """Make traffic light decision based on car counts"""
        if count_a > count_b:
            return "A"
        elif count_b > count_a:
            return "B"
        else:
            # If counts are equal, alternate to prevent starvation
            return "A" if self.current_light != "A" else "B"
    
    def send_decision(self, decision):
        """Send decision to orchestrator"""
        success = self.communicator.send_decision(decision)
        if success:
            self.current_light = decision
            print(f"Decisão enviada: abrir semáforo {decision}")
        else:
            print("Falha ao enviar decisão ao orquestrador")
        return success
    
    def send_status(self, count_a, count_b):
        """Send status update to orchestrator"""
        return self.communicator.send_status(count_a, count_b)
    
    def run_cycle(self):
        """Run one complete decision cycle"""
        # Process both cameras
        count_a = self.process_frame(self.camera_a, self.car_identifier_a, "A")
        count_b = self.process_frame(self.camera_b, self.car_identifier_b, "B")
        
        # Make decision
        decision = self.make_decision(count_a, count_b)
        
        # Send decision and status
        self.send_decision(decision)
        self.send_status(count_a, count_b)
        
        # Update cycle counter
        self.cycle_count += 1
        
        return decision, count_a, count_b
    
    def run_loop(self):
        """Main processing loop"""
        print("Iniciando sistema de controle de semáforos...")
        print(f"Modo: {'Raspberry Pi' if IS_RASPBERRY_PI else 'Desktop'}")
        
        if not self.initialize_cameras():
            return
        
        self.running = True
        self.start_time = time.time()
        
        print(f"Iniciando loop de processamento (intervalo: {PROCESSING_SETTINGS['decision_interval']}s)")
        
        try:
            while self.running:
                cycle_start = time.time()
                
                # Run one complete cycle
                decision, count_a, count_b = self.run_cycle()
                
                # Calculate remaining time for this cycle
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, PROCESSING_SETTINGS['decision_interval'] - cycle_time)
                
                # Log performance every 10 cycles
                if self.cycle_count % 10 == 0:
                    elapsed = time.time() - self.start_time
                    avg_cycle_time = elapsed / self.cycle_count
                    print(f"Ciclo {self.cycle_count}: {count_a} vs {count_b} carros -> Semáforo {decision}")
                    print(f"  Tempo ciclo: {cycle_time:.2f}s, Média: {avg_cycle_time:.2f}s")
                
                # Sleep for remaining cycle time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("\nInterrupção pelo usuário")
        except Exception as e:
            print(f"Erro no loop principal: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Encerrando sistema...")
        self.running = False
        
        # Release cameras
        if self.camera_a:
            self.camera_a.release()
        if self.camera_b:
            self.camera_b.release()
        
        # Print final statistics
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"\nEstatísticas finais:")
            print(f"  Ciclos completados: {self.cycle_count}")
            print(f"  Tempo total: {total_time:.2f}s")
            print(f"  Média de ciclos por minuto: {self.cycle_count * 60 / total_time:.1f}")
            
            avg_a = self.car_identifier_a.get_average_processing_time()
            avg_b = self.car_identifier_b.get_average_processing_time()
            print(f"  Tempo médio processamento - Câmera A: {avg_a:.3f}s, Câmera B: {avg_b:.3f}s")

def main_teste():
    """Test function for development environment"""
    print("=== MODO DE TESTE ===")
    print("Executando teste básico do sistema...")
    
    # Create test controller
    controller = TrafficLightController()
    
    # Test camera initialization
    if controller.initialize_cameras():
        print("✓ Câmeras inicializadas com sucesso")
        
        # Run one test cycle
        print("\nExecutando ciclo de teste...")
        decision, count_a, count_b = controller.run_cycle()
        print(f"✓ Ciclo completo: {count_a} carros na câmera A, {count_b} carros na câmera B")
        print(f"✓ Decisão: abrir semáforo {decision}")
        
    else:
        print("✗ Falha ao inicializar câmeras")
    
    controller.cleanup()
    print("\nTeste concluído!")

def get_env_mode():
    """Get mode from environment variable"""
    mode = os.getenv('MODO', 'production').lower()
    return mode

if __name__ == "__main__":
    modo = get_env_mode()
    
    if modo == "development":
        main_teste()
    else:
        # Production mode - run full traffic light controller
        controller = TrafficLightController(
            camera_a_index=0,
            camera_b_index=1,
            orchestrator_host='localhost',
            orchestrator_port=9000
        )
        controller.run_loop()