import cv2
import os
import time
import threading
import signal
import sys
import numpy as np
from dotenv import load_dotenv
from src.models.car_identify import create_car_identifier
from src.application.comunicator import OrchestratorComunicator
from src.settings.rpi_config import CAMERA_SETTINGS, PROCESSING_SETTINGS, MODEL_SETTINGS, NETWORK_SETTINGS, IS_RASPBERRY_PI

# Load environment variables from .env file
load_dotenv()

# Disable OpenCV logging in production mode
modo = os.getenv('MODO', 'production').lower()
if modo != 'development':
    cv2.setLogLevel(0)  # Disable all OpenCV logging
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    # Suppress stderr to hide OpenCV warnings in production
    import sys
    from contextlib import redirect_stderr
    devnull = open(os.devnull, 'w')
    sys.stderr = devnull
else:
    # In development mode, keep warnings but mark them as debug info
    print("Debug: OpenCV warnings will be displayed")

class TrafficLightController:
    def __init__(self, camera_a_dev='/dev/video0', camera_b_dev='/dev/video2', orchestrator_host='localhost', orchestrator_port=9000):
        self.camera_a_dev = camera_a_dev
        self.camera_b_dev = camera_b_dev
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port

        # Initialize cameras with optimized settings
        self.camera_a = None
        self.camera_b = None

        # Initialize car identifiers with conservative CV detection for reliable counting
        print("🚗 Using conservative CV car detection for reliable multi-car counting...")
        self.car_identifier_a = create_car_identifier('rpi' if IS_RASPBERRY_PI else 'desktop', use_custom_model=False)
        self.car_identifier_b = create_car_identifier('rpi' if IS_RASPBERRY_PI else 'desktop', use_custom_model=False)

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

        # Visualization threads
        self.vis_thread_a = None
        self.vis_thread_b = None
        self.vis_running_a = False
        self.vis_running_b = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nRecebido sinal {signum}. Encerrando...")
        self.running = False
        self.vis_running_a = False
        self.vis_running_b = False
    
    def initialize_cameras(self):
        """Initialize cameras with optimized settings"""
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        try:
            camera_a_index = int(os.getenv('CAMERA_A_INDEX', '0'))
            camera_b_index = int(os.getenv('CAMERA_B_INDEX', '0'))

            # Try to use real cameras first (in both development and production modes)
            print(f"Debug: Attempting to initialize real cameras - A: index {camera_a_index}, B: index {camera_b_index}")

            # Camera A
            self.camera_a = cv2.VideoCapture(camera_a_index)
            if not self.camera_a.isOpened():
                raise Exception(f"Failed to open camera A at index {camera_a_index}")

            # Camera B
            if camera_a_index == camera_b_index:
                self.camera_b = self.camera_a
                print("Debug: Using same camera for A and B")
            else:
                self.camera_b = cv2.VideoCapture(camera_b_index)
                if not self.camera_b.isOpened():
                    raise Exception(f"Failed to open camera B at index {camera_b_index}")

            # Set camera properties
            if IS_RASPBERRY_PI:
                # Raspberry Pi camera optimizations
                if not self.camera_a.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width']):
                    if debug_mode: print("Debug: Failed to set camera A width")
                if not self.camera_a.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height']):
                    if debug_mode: print("Debug: Failed to set camera A height")
                if not self.camera_a.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps']):
                    if debug_mode: print("Debug: Failed to set camera A FPS")
                if not self.camera_a.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_SETTINGS['buffer_size']):
                    if debug_mode: print("Debug: Failed to set camera A buffer size")

                if self.camera_a != self.camera_b:
                    if not self.camera_b.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width']):
                        if debug_mode: print("Debug: Failed to set camera B width")
                    if not self.camera_b.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height']):
                        if debug_mode: print("Debug: Failed to set camera B height")
                    if not self.camera_b.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps']):
                        if debug_mode: print("Debug: Failed to set camera B FPS")
                    if not self.camera_b.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_SETTINGS['buffer_size']):
                        if debug_mode: print("Debug: Failed to set camera B buffer size")
            else:
                # Desktop settings
                if not self.camera_a.set(cv2.CAP_PROP_FRAME_WIDTH, 640):
                    if debug_mode: print("Debug: Failed to set camera A width")
                if not self.camera_a.set(cv2.CAP_PROP_FRAME_HEIGHT, 480):
                    if debug_mode: print("Debug: Failed to set camera A height")
                if not self.camera_a.set(cv2.CAP_PROP_FPS, 15):
                    if debug_mode: print("Debug: Failed to set camera A FPS")

                if self.camera_a != self.camera_b:
                    if not self.camera_b.set(cv2.CAP_PROP_FRAME_WIDTH, 640):
                        if debug_mode: print("Debug: Failed to set camera B width")
                    if not self.camera_b.set(cv2.CAP_PROP_FRAME_HEIGHT, 480):
                        if debug_mode: print("Debug: Failed to set camera B height")
                    if not self.camera_b.set(cv2.CAP_PROP_FPS, 15):
                        if debug_mode: print("Debug: Failed to set camera B FPS")

            # Test cameras
            ret_a, _ = self.camera_a.read()
            ret_b, _ = self.camera_b.read()

            if not ret_a or not ret_b:
                raise Exception("Failed to read from one or both cameras")

            print("Real cameras initialized successfully")
            # Start visualization threads
            self._start_visualization_threads()
            return True

        except Exception as e:
            print(f"Failed to initialize real cameras: {e}")
            print("Falling back to test images...")

            # Fallback to test images
            print("Debug: Development mode detected - using test images")
            # Check if test images exist
            test_image = "src/Data/test_frame.jpg"
            if os.path.exists(test_image):
                print("Debug: Using existing test image")
                self.camera_a = cv2.imread(test_image)
                self.camera_b = cv2.imread(test_image)
                if self.camera_a is None or self.camera_b is None:
                    raise Exception("Failed to load test images")
            else:
                print("Debug: No test image found - trying real images from Data directory")
                # Try to load real images from Data directory
                available_images = [
                    "src/Data/0410.png",
                    "src/Data/carrinho-de-formula-1-de-plastico_120031_600_1.jpg",
                    "src/Data/carrinho_de_friccao_f1_super_racing_com_luz_e_som_dm_toys_26706_1_3a3df60bac059be0a8dabc7bb3972f9f.webp",
                    "src/Data/images.jpeg",
                    "src/Data/images (1).jpeg",
                    "src/Data/images (2).jpeg",
                    "src/Data/D_699956-MLB43150592695_082020-O.jpg"
                ]

                # Try to load real images first
                loaded_a = False
                loaded_b = False

                for img_path in available_images:
                    if os.path.exists(img_path):
                        try:
                            img = cv2.imread(img_path)
                            if img is not None and img.size > 0:
                                # Verificar se imagem foi carregada corretamente
                                print(f"Debug: Image loaded successfully from {img_path} - Shape: {img.shape}, Type: {img.dtype}")

                                if not loaded_a:
                                    self.camera_a = img
                                    loaded_a = True
                                    print(f"Debug: ✅ Loaded real test image A from {img_path}")
                                elif not loaded_b:
                                    self.camera_b = img
                                    loaded_b = True
                                    print(f"Debug: ✅ Loaded real test image B from {img_path}")
                                    break
                            else:
                                print(f"Debug: ❌ Failed to load image from {img_path} - invalid or empty image")
                        except Exception as e:
                            print(f"Debug: ❌ Error loading image from {img_path}: {e}")
                    else:
                        print(f"Debug: File does not exist: {img_path}")

                # Após tentar carregar imagens reais, mostrar status final
                print(f"Debug: Final status - A loaded: {loaded_a}, B loaded: {loaded_b}")
                if loaded_a and loaded_b:
                    print("Debug: 🎯 SUCCESS: Using REAL images from Data directory!")

                # If couldn't load real images, create synthetic ones with detectable objects
                if not loaded_a:
                    self.camera_a = self._create_test_frame_with_objects("A")
                    print("Debug: Using synthetic test frame for camera A")

                if not loaded_b:
                    self.camera_b = self._create_test_frame_with_objects("B")
                    print("Debug: Using synthetic test frame for camera B")

            print("Debug: Test data loaded successfully")
            return True
            
        except Exception as e:
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: Erro ao inicializar câmeras: {e}")
            else:
                print("Erro: Não foi possível inicializar as câmeras.")
            return False
    
    def process_frame(self, camera, car_identifier, camera_name):
        """Process a single frame from a camera"""
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        # Handle development mode with static images
        if not hasattr(camera, 'read'):
            # camera is a static image
            frame = camera.copy()
            ret = frame is not None
        else:
            # Production mode: real camera
            ret, frame = camera.read()

        if not ret or frame is None:
            if debug_mode:
                print(f"Debug: Erro ao capturar frame da câmera {camera_name}")
            return 0

        # Count cars with detailed logging
        if debug_mode:
            print(f"Debug: 🔍 Camera {camera_name} - Iniciando detecção de carros...")
            print(f"Debug: 📐 Frame shape: {frame.shape}")
            print(f"Debug: 🤖 IA/car_identify executando...")

        car_count = car_identifier.count_cars(frame)

        if debug_mode:
            print(f"Debug: ✅ Camera {camera_name} DETECTOU: {car_count} carros")
            if car_count > 0:
                print(f"Debug: 🚗 Fluxo de tráfego ATIVO na direção {camera_name}")

        # Log processing time occasionally
        if car_identifier.frame_count % 30 == 0:
            avg_time = car_identifier.get_average_processing_time()
            print(f"Câmera {camera_name}: {car_count} carros detectados (tempo médio: {avg_time:.3f}s)")

        return car_count
    
    def make_decision(self, count_a, count_b):
        """Make traffic light decision based on car counts - prioritize higher traffic flow"""
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        if debug_mode:
            print(f"Debug: ⚖️  DECISÃO DE SEMÁFORO:")
            print(f"Debug: 👈 Direção A tem: {count_a} carros")
            print(f"Debug: 👉 Direção B tem: {count_b} carros")

        if count_a > count_b:
            if debug_mode:
                print(f"Debug: 🏆 PRIORIDADE PARA A - MAIS fluxo de carros ({count_a} > {count_b})")
            return "A"  # 🟢 Prioridade para A - MAIS carros passando
        elif count_b > count_a:
            if debug_mode:
                print(f"Debug: 🏆 PRIORIDADE PARA B - MAIS fluxo de carros ({count_b} > {count_a})")
            return "B"  # 🟢 Prioridade para B - MAIS carros passando
        else:
            # Se contagens iguais, alterna para evitar fome
            alternate_to = "A" if self.current_light != "A" else "B"
            if debug_mode:
                print(f"Debug: ⚖️  Fluxos IGUAIS - Alternando para {alternate_to} (evita fome)")
            return alternate_to
    
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
    
    def _create_test_frame_with_objects(self, direction):
        """Create a test frame with objects that can be detected"""
        # Create base frame
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some background variation (not completely black/white)
        frame[:, :] = [50, 100, 50]  # Dark green background (like grass/road)

        # Add rectangular objects that look like cars/trucks
        if direction == "A":
            # Add 2 "cars" with different sizes
            cv2.rectangle(frame, (100, 200), (200, 300), (100, 100, 200), -1)  # Car 1
            cv2.rectangle(frame, (300, 180), (400, 280), (150, 150, 100), -1)  # Car 2
        else:
            # Add 1 "car" for camera B
            cv2.rectangle(frame, (150, 220), (280, 350), (200, 100, 100), -1)  # Car

        # Add some noise/variation to make it more realistic
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        return frame.astype(np.uint8)

    def _has_display(self):
        """Check if graphical display is available"""
        return 'DISPLAY' in os.environ and os.environ['DISPLAY']

    def _start_visualization_threads(self):
        """Start background threads for real-time camera visualization"""
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        if debug_mode:
            print("Debug: Starting visualization threads for real-time display")

        # Initialize flags for headless mode
        if not self._has_display():
            print("Debug: No display detected - saving visualization frames to images")
            self.frame_saved_a = False
            self.frame_saved_b = False

        # Start threads based on unique cameras
        if self.camera_a is self.camera_b:
            # Same camera for A and B
            self.vis_running_a = True
            self.vis_thread_a = threading.Thread(
                target=self._visualization_loop,
                args=(self.camera_a, self.car_identifier_a, "A/B"),
                daemon=True
            )
            self.vis_thread_a.start()
            print("Debug: Thread for shared camera A/B started")
        else:
            # Different cameras
            self.vis_running_a = True
            self.vis_thread_a = threading.Thread(
                target=self._visualization_loop,
                args=(self.camera_a, self.car_identifier_a, "A"),
                daemon=True
            )
            self.vis_thread_a.start()
            print("Debug: Thread A started")

            self.vis_running_b = True
            self.vis_thread_b = threading.Thread(
                target=self._visualization_loop,
                args=(self.camera_b, self.car_identifier_b, "B"),
                daemon=True
            )
            self.vis_thread_b.start()
            print("Debug: Thread B started")

        import sys
        sys.stdout.flush()

    def _visualization_loop(self, camera, car_identifier, camera_name):
        """Continuous real-time visualization of camera feed with detections"""
        print(f"Debug: Continuous visualization loop started for {camera_name}")
        import sys
        sys.stdout.flush()
        import time
        start_time = time.time()
        window_name = f"Detecção IA - Câmera {camera_name}"
        has_display = self._has_display()
        frame_count = 0

        if has_display:
            print(f"Debug: Starting continuous live feed for {camera_name}. Press 'ESC' to close.")
            cv2.namedWindow(window_name)

        while self.vis_running_a if camera_name in ["A", "A/B"] else self.vis_running_b:
            try:
                # Capture frame
                if hasattr(camera, 'read'):
                    ret, frame = camera.read()
                else:
                    # For test images in development mode
                    frame = camera.copy()
                    ret = frame is not None

                if not ret or frame is None:
                    time.sleep(0.1)
                    continue

                # Run detection and visualization
                car_count, vis_frame = car_identifier.visualize_detection(frame, show_contours=True)

                # Add camera info
                h, w = vis_frame.shape[:2]
                model_status = "🤖 IA ML" if car_identifier.model_loaded else "🔍 CV TRADICIONAL"

                # Background for text
                cv2.rectangle(vis_frame, (10, 10), (w-10, 60), (0, 0, 0), -1)
                cv2.rectangle(vis_frame, (10, 10), (w-10, 60), (255, 255, 255), 2)

                # Title and info
                cv2.putText(vis_frame, f"Câmera {camera_name} - {model_status}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(vis_frame, f"Detecção em Tempo Real - {car_count} veículo(s)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                # Instructions
                if has_display:
                    cv2.putText(vis_frame, "PRESSIONE 'ESC' para fechar",
                                (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(vis_frame, "Modo sem display - salvando video",
                                (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if has_display:
                    # Show window continuously
                    try:
                        cv2.imshow(window_name, vis_frame)
                    except Exception as e:
                        print(f"Debug: Failed to show window {window_name}: {e}")
                        break

                    # Check if window was closed by user (e.g., clicking X button)
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 0:
                        print(f"Debug: Window {window_name} was closed by user")
                        if camera_name in ["A", "A/B"]:
                            self.vis_running_a = False
                        else:
                            self.vis_running_b = False
                        break

                    # Handle key presses continuously
                    key = cv2.waitKey(1)  # Check for key press without delay
                    if key == 27:  # ESC key
                        print(f"Debug: ESC pressed for {camera_name}, closing window")
                        if camera_name in ["A", "A/B"]:
                            self.vis_running_a = False
                        else:
                            self.vis_running_b = False
                        break
                else:
                    # Headless mode: save frame to image (once for static images)
                    vis_frame_resized = cv2.resize(vis_frame, (640, 480))

                    if not getattr(self, f'frame_saved_{camera_name.lower()}', False):
                        output_path = f'camera_{camera_name}_frame.jpg'
                        try:
                            success = cv2.imwrite(output_path, vis_frame_resized)
                            if success:
                                print(f"Debug: Frame saved successfully for camera {camera_name} at {output_path}")
                                setattr(self, f'frame_saved_{camera_name.lower()}', True)
                            else:
                                print(f"Debug: Failed to save frame for camera {camera_name}")
                        except Exception as e:
                            print(f"Debug: Error saving frame for camera {camera_name}: {e}")

                    # Print status occasionally
                    frame_count += 1
                    if frame_count % 30 == 0:  # Every ~3 seconds at 10 FPS
                        print(f"Câmera {camera_name}: {car_count} veículo(s) detectado(s)")

                    time.sleep(0.1)  # Control loop speed

            except Exception as e:
                print(f"Erro na visualização da câmera {camera_name}: {e}")
                time.sleep(0.5)

        elapsed = time.time() - start_time
        print(f"Debug: Continuous visualization for camera {camera_name} ran for {elapsed:.2f} seconds")

        if has_display:
            cv2.destroyWindow(window_name)

    def _create_visualization(self, frame, car_identifier, camera_name, car_count):
        """Create visualization of car detection and save annotated frames periodically"""
        try:
            debug_mode = os.getenv('MODO', '').lower() == 'development'

            # Create visualization with bounding boxes
            car_count_vis, vis_frame_with_boxes = car_identifier.visualize_detection(frame, show_contours=True)

            # Save annotated frame every cycle for visual inspection
            output_dir = "detection_frames"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{output_dir}/camera_{camera_name}_cycle_{self.cycle_count}.jpg"
            cv2.imwrite(filename, vis_frame_with_boxes)
            if debug_mode:
                print(f"Debug: 📸 Frame anotado salvo: {filename}")

        except Exception as e:
            if debug_mode:
                print(f"Debug: Erro na criação de visualização: {e}")
            else:
                print(f"Erro na criação de visualização: {e}")

    def _close_windows_on_user_input(self):
        """Allow user to close windows gracefully"""
        print("\n📱 Janelas gráficas abertas!")
        print("💡 Pressione Ctrl+C novamente para fechar tudo")
        print("⏰ Ou aguarde alguns segundos...")

        # Give user time to see the content
        import time
        time.sleep(3)

        # Auto-close after short delay
        cv2.destroyAllWindows()
        print("✅ Janelas gráficas fechadas")

    def cleanup(self):
        """Clean up resources"""
        print("\n🔄 Encerrando sistema...")

        # Stop visualization threads
        self.vis_running_a = False
        self.vis_running_b = False
        if self.vis_thread_a and self.vis_thread_a.is_alive():
            self.vis_thread_a.join(timeout=1.0)
        if self.vis_thread_b and self.vis_thread_b.is_alive():
            self.vis_thread_b.join(timeout=1.0)

        # Note: Frames saved to images in headless mode

        # Avisar sobre janelas - verificar se existem janelas abertas
        try:
            # Check if any windows exist by trying to access one
            active_windows = cv2.getWindowProperty("Detecção IA - Câmera A", cv2.WND_PROP_VISIBLE)
            if active_windows >= 0:  # Window exists and is visible
                print("❌ Fechando janela(s) gráfica(s)...")
        except:
            # No windows to close
            pass

        # Force close all windows
        cv2.destroyAllWindows()

        self.running = False

        # Release cameras - handle both real cameras and test images
        # Release if they have release method (real cameras)
        if self.camera_a and hasattr(self.camera_a, 'release'):
            self.camera_a.release()
        if self.camera_b and hasattr(self.camera_b, 'release') and self.camera_b != self.camera_a:
            self.camera_b.release()
        print("Debug: Cameras released")

        print("✅ Sistema encerrado com sucesso!")

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
        print("📺 Janelas de visualização abertas. Pressione 'ESC' nas janelas para fechar.")

        # Keep visualization running until all windows are closed or ESC is pressed
        try:
            while controller.vis_running_a or controller.vis_running_b:
                time.sleep(0.1)  # Keep main thread alive
        except KeyboardInterrupt:
            print("\nInterrupção pelo usuário")

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
            camera_a_dev='/dev/video0',
            camera_b_dev='/dev/video0',
            orchestrator_host='localhost',
            orchestrator_port=9000
        )
        controller.run_loop()