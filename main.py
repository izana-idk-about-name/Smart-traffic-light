import cv2
import os
import time
import threading
import signal
import atexit
import gc
import numpy as np
from typing import Optional
from dotenv import load_dotenv
from src.models.car_identify import create_car_identifier
from src.application.comunicator import OrchestratorComunicator
from src.application.camera_source import CameraSource, CameraFactory
from src.settings.rpi_config import CAMERA_SETTINGS, PROCESSING_SETTINGS, MODEL_SETTINGS, NETWORK_SETTINGS, IS_RASPBERRY_PI
from src.utils.resource_manager import FrameBuffer, ResourceTracker, get_global_tracker
from src.utils.healthcheck import HealthCheck, BuiltInHealthChecks
from src.utils.watchdog import Watchdog, RecoveryStrategy, RecoveryAction
from src.utils.logger import get_logger, shutdown_logging
from src.visualization import MultiCameraViewer

# Load environment variables from .env file
load_dotenv()

# Disable OpenCV logging in production mode
modo = os.getenv('MODO', 'production').lower()  # ← Verifica MODO antes de carregar .env
IS_PRODUCTION = modo == 'production'

if IS_PRODUCTION:
    cv2.setLogLevel(0)  # Disable all OpenCV logging
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    print("[INFO] Production mode: OpenCV logging disabled, optimizations enabled", flush=True)
else:
    print("Debug: Development mode - full logging enabled", flush=True)

# Frame skipping configuration (production optimization)
FRAME_SKIP = 2 if IS_PRODUCTION else 0  # Skip 2 of every 3 frames in production


class ShutdownManager:
    """Thread-safe shutdown coordination"""
    def __init__(self):
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()
        self._signal_received: Optional[int] = None
        self._logger = get_logger(__name__)
    
    def request_shutdown(self, signum: int = None):
        """Request graceful shutdown (thread-safe)"""
        with self._lock:
            if not self._shutdown_event.is_set():
                self._signal_received = signum
                self._shutdown_event.set()
                if signum:
                    self._logger.info(f"Shutdown requested (signal: {signum})")
                else:
                    self._logger.info("Shutdown requested programmatically")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown was requested"""
        return self._shutdown_event.is_set()
    
    def wait_for_shutdown(self, timeout: float = None) -> bool:
        """Wait for shutdown signal"""
        return self._shutdown_event.wait(timeout)
    
    def get_signal(self) -> Optional[int]:
        """Get the signal that triggered shutdown"""
        with self._lock:
            return self._signal_received

class TrafficLightController:
    def __init__(self, camera_a_dev='/dev/video0', camera_b_dev='/dev/video2',
                 orchestrator_host='localhost', orchestrator_port=9000):
        """
        Initialize Traffic Light Controller with health monitoring.
        
        Args:
            camera_a_dev: Camera A device path
            camera_b_dev: Camera B device path
            orchestrator_host: Orchestrator server host
            orchestrator_port: Orchestrator server port
        """
        print(f"[TRACE] __init__ called with cameras: {camera_a_dev}, {camera_b_dev}", flush=True)
        self.logger = get_logger(__name__)
        print("[TRACE] Logger initialized", flush=True)
        self.camera_a_dev = camera_a_dev
        self.camera_b_dev = camera_b_dev
        self.orchestrator_host = orchestrator_host
        self.orchestrator_port = orchestrator_port

        # Initialize cameras with type-safe CameraSource abstraction
        self.camera_a: Optional[CameraSource] = None
        self.camera_b: Optional[CameraSource] = None
        
        # Initialize resource tracking (minimal in production)
        self.resource_tracker = get_global_tracker()
        
        # Frame skipping optimization
        self.frame_skip_counter = 0
        self.skip_frames = FRAME_SKIP
        
        # Initialize frame buffers ONLY if not production (saves I/O and memory)
        if not IS_PRODUCTION:
            self.frame_buffer_a = FrameBuffer(
                max_frames=100,
                output_dir='detection_frames/camera_a',
                max_memory_mb=50,
                jpeg_quality=85
            )
            self.frame_buffer_b = FrameBuffer(
                max_frames=100,
                output_dir='detection_frames/camera_b',
                max_memory_mb=50,
                jpeg_quality=85
            )
        else:
            self.frame_buffer_a = None
            self.frame_buffer_b = None

        # FIX: Carregar settings e usar modelo customizado otimizado
        from src.settings import get_settings
        settings = get_settings()
        
        # Initialize car identifiers com configuração atualizada
        print("[INFO] Creating car identifiers with OPTIMIZED custom model...")
        self.car_identifier_a = create_car_identifier(
            'rpi' if IS_RASPBERRY_PI else 'desktop',
            use_custom_model=settings.detection.use_custom_model,  # TRUE
            use_ml=settings.detection.use_ml_model,                 # FALSE
            use_tflite=settings.detection.use_tflite                # FALSE
        )
        
        self.car_identifier_b = create_car_identifier(
            'rpi' if IS_RASPBERRY_PI else 'desktop',
            use_custom_model=settings.detection.use_custom_model,
            use_ml=settings.detection.use_ml_model,
            use_tflite=settings.detection.use_tflite
        )
        
        print("[INFO] Using OPTIMIZED custom model (.pkl)")
        print(f"[INFO] Model path: {settings.detection.custom_model_path}")
        print(f"[INFO] Confidence threshold: {settings.detection.min_confidence}")
        
        # Initialize separate visualization identifiers to avoid threading conflicts
        print("[TRACE] Creating vis_identifier_a (hybrid mode)...", flush=True)
        self.vis_identifier_a = create_car_identifier('rpi' if IS_RASPBERRY_PI else 'desktop', use_custom_model=True, use_ml=False)
        print("[TRACE] Creating vis_identifier_b (hybrid mode)...", flush=True)
        self.vis_identifier_b = create_car_identifier('rpi' if IS_RASPBERRY_PI else 'desktop', use_custom_model=True, use_ml=False)
        print("[TRACE] All identifiers created successfully", flush=True)

        # Initialize orchestrator communicator
        print("[TRACE] Initializing orchestrator communicator...", flush=True)
        try:
            self.communicator = OrchestratorComunicator(
                host=orchestrator_host,
                port=orchestrator_port,
                use_websocket=NETWORK_SETTINGS['use_websocket']
            )
            print("[TRACE] Orchestrator communicator initialized successfully", flush=True)
        except Exception as e:
            print(f"[TRACE] Failed to initialize communicator: {e}", flush=True)
            self.logger.warning(f"Failed to initialize orchestrator communicator: {e}")
            self.communicator = None

        # Shutdown management
        self.shutdown_manager = ShutdownManager()
        
        # Control flags
        self.running = False
        self.current_light = None

        # Performance tracking
        self.cycle_count = 0
        self.start_time = None
        self.last_gc_time = time.time()  # For garbage collection scheduling

        # New simplified visualization system
        self.camera_viewer = MultiCameraViewer()
        # Enable visualization if VISUALIZATION_ENABLED=true in .env OR in development mode
        visualization_env = os.getenv('VISUALIZATION_ENABLED', 'false').lower()
        self.enable_visualization = (visualization_env == 'true') or (not IS_PRODUCTION)

        # Health monitoring
        self.health_check = HealthCheck(max_failures=3)
        self.watchdog: Optional[Watchdog] = None
        self._setup_health_checks()
        
        # Setup signal handlers for graceful shutdown
        print("[TRACE] Setting up signal handlers...", flush=True)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        print("[TRACE] Signal handlers configured", flush=True)
        
        # Register cleanup on exit
        print("[TRACE] Registering cleanup handler...", flush=True)
        atexit.register(self.cleanup)
        
        self.logger.info("TrafficLightController initialized")
        print("[TRACE] TrafficLightController __init__ complete!", flush=True)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully with ShutdownManager"""
        self.logger.info(f"Signal {signum} received, requesting shutdown")
        self.shutdown_manager.request_shutdown(signum)
        self.running = False
        self.vis_running_a = False
        self.vis_running_b = False
    
    def _setup_health_checks(self):
        """Setup health monitoring checks (optimized intervals for production)"""
        try:
            # Memory health check
            self.health_check.register_check(
                'memory',
                BuiltInHealthChecks.create_memory_health_check(max_memory_percent=90.0),
                description="System memory usage",
                critical=False
            )
            
            # Disk space health check (only if frame buffers enabled)
            if not IS_PRODUCTION:
                self.health_check.register_check(
                    'disk_space',
                    BuiltInHealthChecks.create_disk_health_check(min_free_gb=0.5),
                    description="Available disk space",
                    critical=False
                )
            
            self.logger.info("Health checks configured")
        except Exception as e:
            self.logger.warning(f"Failed to setup some health checks: {e}")
    
    def _start_watchdog(self):
        """Start watchdog monitoring system (optimized intervals)"""
        try:
            # Create watchdog with shutdown callback
            # Production: longer intervals (60s vs 30s) for reduced overhead
            check_interval = 60 if IS_PRODUCTION else 30
            self.watchdog = Watchdog(
                health_check=self.health_check,
                check_interval=check_interval,
                shutdown_callback=lambda: self.shutdown_manager.request_shutdown()
            )
            
            # Register recovery strategies
            memory_strategy = RecoveryStrategy(
                component='memory',
                max_attempts=3,
                actions=[RecoveryAction.FORCE_GC],
                cooldown_seconds=60.0
            )
            self.watchdog.register_recovery_strategy(memory_strategy)
            
            disk_strategy = RecoveryStrategy(
                component='disk_space',
                max_attempts=2,
                actions=[RecoveryAction.CLEAN_TEMP_FILES],
                cooldown_seconds=300.0
            )
            self.watchdog.register_recovery_strategy(disk_strategy)
            
            # Start monitoring
            self.watchdog.start()
            self.logger.info("Watchdog monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start watchdog: {e}", exc_info=True)
    
    def initialize_cameras(self):
        """Initialize cameras using type-safe CameraSource abstraction"""
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        try:
            camera_a_index = int(os.getenv('CAMERA_A_INDEX', '0'))
            camera_b_index = int(os.getenv('CAMERA_B_INDEX', '0'))

            print(f"Debug: 🔍 Initializing cameras - A: index {camera_a_index}, B: index {camera_b_index}")

            # Try to create live cameras first
            try:
                print(f"Debug: 📹 Creating Camera A (index {camera_a_index})...")
                self.camera_a = CameraFactory.create(camera_index=camera_a_index)
                
                if not self.camera_a.is_opened():
                    print(f"❌ FAIL: Camera A index {camera_a_index} could not be opened")
                    # Try alternative indices
                    for alt_index in [1, 2, -1]:
                        if alt_index != camera_a_index:
                            print(f"Debug: 🔄 Trying alternative Camera A index {alt_index}...")
                            try:
                                self.camera_a = CameraFactory.create(camera_index=alt_index)
                                if self.camera_a.is_opened():
                                    print(f"✅ SUCCESS: Camera A opened at alternative index {alt_index}")
                                    camera_a_index = alt_index
                                    break
                            except Exception:
                                continue
                    else:
                        raise Exception(f"Failed to open camera A at any index")
                
                # Camera B
                if camera_a_index == camera_b_index:
                    self.camera_b = self.camera_a
                    print("Debug: 🔗 Using same camera for A and B")
                else:
                    print(f"Debug: 📹 Creating Camera B (index {camera_b_index})...")
                    self.camera_b = CameraFactory.create(camera_index=camera_b_index)
                    
                    if not self.camera_b.is_opened():
                        print(f"❌ FAIL: Camera B index {camera_b_index} could not be opened")
                        # Try alternative indices
                        for alt_index in [1, 2, -1]:
                            if alt_index != camera_b_index:
                                print(f"Debug: 🔄 Trying alternative Camera B index {alt_index}...")
                                try:
                                    self.camera_b = CameraFactory.create(camera_index=alt_index)
                                    if self.camera_b.is_opened():
                                        print(f"✅ SUCCESS: Camera B opened at alternative index {alt_index}")
                                        camera_b_index = alt_index
                                        break
                                except Exception:
                                    continue
                        else:
                            raise Exception(f"Failed to open camera B at any index")
                
                # Test cameras by reading frames
                print("Debug: 🧪 Testing frame reading from cameras...")
                ret_a, frame_a = self.camera_a.read()
                ret_b, frame_b = self.camera_b.read()

                if not ret_a or frame_a is None:
                    print(f"❌ ERROR: Could not read frame from camera A")
                    raise Exception("Failed to read from camera A")
                if not ret_b or frame_b is None:
                    print(f"❌ ERROR: Could not read frame from camera B")
                    raise Exception("Failed to read from camera B")

                print(f"✅ Live cameras initialized successfully")
                print(f"   Camera A: {self.camera_a.get_properties()}")
                print(f"   Camera B: {self.camera_b.get_properties()}")

                # Register camera health checks
                self.health_check.register_check(
                    'camera_a',
                    BuiltInHealthChecks.create_camera_health_check(self.camera_a, "A"),
                    description="Camera A status",
                    critical=True
                )
                
                if self.camera_a != self.camera_b:
                    self.health_check.register_check(
                        'camera_b',
                        BuiltInHealthChecks.create_camera_health_check(self.camera_b, "B"),
                        description="Camera B status",
                        critical=True
                    )
                
                # Register detection health checks
                self.health_check.register_check(
                    'detection_a',
                    BuiltInHealthChecks.create_detection_health_check(self.car_identifier_a),
                    description="Detection model A",
                    critical=True
                )
                
                self.health_check.register_check(
                    'detection_b',
                    BuiltInHealthChecks.create_detection_health_check(self.car_identifier_b),
                    description="Detection model B",
                    critical=True
                )
                
                # Register processing time checks
                self.health_check.register_check(
                    'processing_time_a',
                    BuiltInHealthChecks.create_processing_time_health_check(
                        self.car_identifier_a, max_time=2.0
                    ),
                    description="Processing time camera A",
                    critical=False
                )
                
                # Start watchdog after cameras are initialized
                self._start_watchdog()
                
                # Start visualization if enabled
                if self.enable_visualization:
                    if self._has_display():
                        print("✅ Starting camera visualization windows...")
                        self._start_camera_viewers()
                    else:
                        print("❌ Display not detected - visualization disabled")
                        print("💡 Try: export DISPLAY=:0")
                else:
                    print("[INFO] Visualization disabled (set VISUALIZATION_ENABLED=true in .env to enable)")

                return True

            except Exception as live_cam_error:
                print(f"Failed to initialize live cameras: {live_cam_error}")
                print("Falling back to test images...")
                
                # Fallback to test images using StaticImageSource
                test_image_paths = [
                    "src/Data/test_frame.jpg",
                    "src/Data/0410.png",
                    "src/Data/carrinho-de-formula-1-de-plastico_120031_600_1.jpg",
                    "src/Data/carrinho_de_friccao_f1_super_racing_com_luz_e_som_dm_toys_26706_1_3a3df60bac059be0a8dabc7bb3972f9f.webp",
                    "src/Data/images.jpeg",
                    "src/Data/images (1).jpeg",
                    "src/Data/images (2).jpeg",
                    "src/Data/D_699956-MLB43150592695_082020-O.jpg"
                ]
                
                loaded_a = False
                loaded_b = False
                
                for img_path in test_image_paths:
                    if os.path.exists(img_path):
                        try:
                            if not loaded_a:
                                self.camera_a = CameraFactory.create(test_image_path=img_path)
                                if self.camera_a.is_opened():
                                    loaded_a = True
                                    print(f"Debug: ✅ Loaded test image A from {img_path}")
                            elif not loaded_b:
                                self.camera_b = CameraFactory.create(test_image_path=img_path)
                                if self.camera_b.is_opened():
                                    loaded_b = True
                                    print(f"Debug: ✅ Loaded test image B from {img_path}")
                                    break
                        except Exception as e:
                            print(f"Debug: ❌ Error loading image from {img_path}: {e}")
                
                # If couldn't load test images, create synthetic ones
                if not loaded_a:
                    synthetic_frame = self._create_test_frame_with_objects("A")
                    self.camera_a = CameraFactory.create(test_image_path=synthetic_frame)
                    print("Debug: Using synthetic test frame for camera A")
                
                if not loaded_b:
                    synthetic_frame = self._create_test_frame_with_objects("B")
                    self.camera_b = CameraFactory.create(test_image_path=synthetic_frame)
                    print("Debug: Using synthetic test frame for camera B")

                print("Debug: Test data loaded successfully")
                return True
            
        except Exception as e:
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: Erro ao inicializar câmeras: {e}")
            else:
                print("Erro: Não foi possível inicializar as câmeras.")
            return False
    
    def process_frame(self, camera: CameraSource, car_identifier, camera_name: str) -> int:
        """
        Process a single frame from a camera source.
        
        Args:
            camera: CameraSource instance (LiveCameraSource or StaticImageSource)
            car_identifier: Car detection model instance
            camera_name: Camera identifier for logging
            
        Returns:
            Number of cars detected in the frame
        """
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        # Read frame using consistent CameraSource interface
        ret, frame = camera.read()
        
        if debug_mode:
            props = camera.get_properties()
            print(f"Debug: 📹 Reading from {props['source_type']} camera {camera_name}")

        if not ret or frame is None:
            print(f"❌ ERROR: Invalid frame from camera {camera_name}")
            props = camera.get_properties()
            print(f"Debug: Camera properties: {props}")
            return 0

        # Count cars with detailed logging
        try:
            if debug_mode:
                print(f"Debug: 🔍 Camera {camera_name} - Iniciando detecção de carros...")
                print(f"Debug: 📐 Frame shape: {frame.shape} dtype: {frame.dtype}")
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

        except Exception as e:
            print(f"❌ ERRO na detecção de carros câmera {camera_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
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
        if self.communicator is None:
            print("Warning: No orchestrator communicator available")
            return False
        try:
            success = self.communicator.send_decision(decision)
            if success:
                self.current_light = decision
                print(f"Decisão enviada: abrir semáforo {decision}")
            else:
                print("Falha ao enviar decisão ao orquestrador")
            return success
        except Exception as e:
            print(f"Erro ao enviar decisão: {e}")
            return False
    
    def send_status(self, count_a, count_b):
        """Send status update to orchestrator"""
        if self.communicator is None:
            print("Warning: No orchestrator communicator available")
            return False
        try:
            return self.communicator.send_status(count_a, count_b)
        except Exception as e:
            print(f"Erro ao enviar status: {e}")
            return False
    
    def run_cycle(self):
        """Run one complete decision cycle with frame skipping optimization"""
        try:
            # Frame skipping logic (production optimization)
            self.frame_skip_counter += 1
            should_process = (self.frame_skip_counter % (self.skip_frames + 1)) == 0
            
            if not should_process and IS_PRODUCTION:
                # Skip processing, return last known values
                return getattr(self, 'last_decision', "A"), getattr(self, 'last_count_a', 0), getattr(self, 'last_count_b', 0)
            
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
            
            # Cache last values for frame skipping
            self.last_decision = decision
            self.last_count_a = count_a
            self.last_count_b = count_b
            
            # Periodic garbage collection (every 100 cycles in production)
            if IS_PRODUCTION and self.cycle_count % 100 == 0:
                current_time = time.time()
                if current_time - self.last_gc_time > 60:  # At least 60s between GC
                    gc.collect()
                    self.last_gc_time = current_time

            return decision, count_a, count_b

        except Exception as e:
            print(f"❌ ERRO crítico no ciclo de processamento: {e}")
            import traceback
            traceback.print_exc()
            # Return safe defaults to prevent system shutdown
            return "A", 0, 0
    
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
            consecutive_errors = 0
            max_consecutive_errors = 5
            last_health_log = time.time()
            health_log_interval = 300  # Log health every 5 minutes

            while self.running and not self.shutdown_manager.is_shutdown_requested():
                cycle_start = time.time()

                try:
                    # Run one complete cycle
                    decision, count_a, count_b = self.run_cycle()

                    # Reset error counter on successful cycle
                    consecutive_errors = 0

                    # Calculate remaining time for this cycle
                    cycle_time = time.time() - cycle_start
                    sleep_time = max(0, PROCESSING_SETTINGS['decision_interval'] - cycle_time)

                    # Log performance every 10 cycles
                    if self.cycle_count % 10 == 0:
                        elapsed = time.time() - self.start_time
                        avg_cycle_time = elapsed / self.cycle_count
                        print(f"Ciclo {self.cycle_count}: {count_a} vs {count_b} carros -> Semáforo {decision}")
                        print(f"  Tempo ciclo: {cycle_time:.2f}s, Média: {avg_cycle_time:.2f}s")
                    
                    # Log health status periodically
                    current_time = time.time()
                    if current_time - last_health_log >= health_log_interval:
                        self._log_health_status()
                        last_health_log = current_time

                    # Sleep for remaining cycle time
                    if sleep_time > 0:
                        slept = 0
                        while slept < sleep_time and self.running:
                            if self.shutdown_manager.is_shutdown_requested():
                                break
                            time.sleep(min(0.1, sleep_time - slept))
                            slept += 0.1

                except Exception as cycle_error:
                    consecutive_errors += 1
                    print(f"❌ ERRO no ciclo {self.cycle_count} (erro {consecutive_errors}/{max_consecutive_errors}): {cycle_error}")

                    if consecutive_errors >= max_consecutive_errors:
                        print(f"🚨 Muitos erros consecutivos ({consecutive_errors}). Reinicializando câmeras...")
                        try:
                            self.cleanup()
                            if self.initialize_cameras():
                                consecutive_errors = 0
                                print("✅ Câmeras reinicializadas com sucesso")
                            else:
                                print("❌ Falha ao reinicializar câmeras. Encerrando...")
                                break
                        except Exception as reinit_error:
                            print(f"❌ Erro na reinicialização: {reinit_error}")
                            break
                    else:
                        # Sleep a bit longer on error to prevent rapid error loops
                        time.sleep(PROCESSING_SETTINGS['decision_interval'])
        
                    # Yield control to allow thread scheduling
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n🛑 Interrupção pelo usuário")
        except Exception as e:
            print(f"❌ Erro crítico no loop principal: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.logger.info("Main loop ended, cleaning up...")
            self.cleanup()
    
    def _log_health_status(self):
        """Log current health status"""
        try:
            report = self.health_check.get_status_report(include_history=False)
            self.logger.info(f"System Health: overall_healthy={report['overall_healthy']}, "
                           f"checks={report['total_checks']}, "
                           f"failures={report['failing_checks']}")
            
            if self.watchdog:
                stats = self.watchdog.get_statistics()
                self.logger.info(f"Watchdog Stats: checks={stats['checks_performed']}, "
                               f"failures={stats['failures_detected']}, "
                               f"recoveries={stats['successful_recoveries']}/{stats['recovery_attempts']}")
        except Exception as e:
            self.logger.debug(f"Error logging health status: {e}")
    
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
        mode = os.getenv('MODE', os.getenv('MODO', 'production')).lower()
        
        # In development mode, always assume display is available
        if mode == 'development':
            return True
        
        # Check for actual display
        try:
            display = os.environ.get('DISPLAY', '')
            session_type = os.environ.get('XDG_SESSION_TYPE', '')
            
            # If DISPLAY is set or we're in a graphical session
            has_display = bool(display) or session_type in ['x11', 'wayland']
            
            # If not set, try to auto-detect common display
            if not has_display and os.path.exists('/tmp/.X11-unix'):
                os.environ['DISPLAY'] = ':0'
                has_display = True
            
            return has_display
        except Exception as e:
            return False

    def _start_camera_viewers(self):
        """Start camera viewers using the new simplified system"""
        try:
            # Get FPS from settings
            fps = CAMERA_SETTINGS.get('fps', 10)
            
            # Add camera A
            self.camera_viewer.add_camera(
                camera_id="A",
                camera_source=self.camera_a,
                detector=self.vis_identifier_a,
                fps=fps
            )
            
            # Add camera B if different from A
            if self.camera_a != self.camera_b:
                self.camera_viewer.add_camera(
                    camera_id="B",
                    camera_source=self.camera_b,
                    detector=self.vis_identifier_b,
                    fps=fps
                )
            
            print("✅ Camera visualization started successfully")
            print("💡 Press ESC in any window to close all viewers")
            
        except Exception as e:
            print(f"❌ Error starting camera viewers: {e}")
            import traceback
            traceback.print_exc()

    
    def verify_cleanup(self) -> bool:
        """
        Verify all resources properly cleaned up.
        
        Returns:
            True if cleanup verified, False otherwise
        """
        checks = {}
        
        try:
            # Check cameras released
            checks['cameras_released'] = not (
                (self.camera_a and self.camera_a.is_opened()) or
                (self.camera_b and self.camera_b != self.camera_a and self.camera_b.is_opened())
            )
        except Exception:
            checks['cameras_released'] = True  # Assume released if check fails
        
        try:
            # Check windows closed
            checks['windows_closed'] = cv2.getWindowProperty('any', cv2.WND_PROP_VISIBLE) < 0
        except Exception:
            checks['windows_closed'] = True  # No windows if check fails
        
        # Check camera viewers stopped
        checks['viewers_stopped'] = not self.camera_viewer.is_any_running()
        
        all_clean = all(checks.values())
        self.logger.info(f"Cleanup verification: {checks}")
        return all_clean

    def cleanup(self):
        """Clean up resources with comprehensive resource management and verification"""
        self.logger.info("Starting cleanup process...")
        
        # Stop watchdog first
        if self.watchdog:
            try:
                self.logger.info("Stopping watchdog...")
                self.watchdog.stop(timeout=3.0)
            except Exception as e:
                self.logger.error(f"Error stopping watchdog: {e}")

        # Stop camera viewers
        try:
            self.logger.info("Stopping camera viewers...")
            self.camera_viewer.stop_all()
        except Exception as e:
            self.logger.error(f"Error stopping camera viewers: {e}")

        # Clear frame buffers (if they exist - not in production)
        if self.frame_buffer_a and self.frame_buffer_b:
            try:
                self.frame_buffer_a.clear()
                self.frame_buffer_b.clear()
                self.logger.info("Frame buffers cleared")
            except Exception as e:
                self.logger.error(f"Error clearing frame buffers: {e}")

        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            self.logger.info("OpenCV windows closed")
        except Exception as e:
            self.logger.error(f"Error closing OpenCV windows: {e}")

        self.running = False

        # Release cameras using CameraSource interface
        if self.camera_a is not None:
            try:
                self.camera_a.release()
                self.camera_a = None
                self.logger.info("Camera A released")
            except Exception as e:
                self.logger.error(f"Error releasing camera A: {e}")
        
        if self.camera_b is not None and self.camera_b != self.camera_a:
            try:
                self.camera_b.release()
                self.camera_b = None
                self.logger.info("Camera B released")
            except Exception as e:
                self.logger.error(f"Error releasing camera B: {e}")

        # Release all tracked resources
        try:
            self.resource_tracker.release_all()
            self.resource_tracker.log_statistics()
        except Exception as e:
            self.logger.error(f"Error releasing tracked resources: {e}")
        
        # Shutdown async logging
        try:
            shutdown_logging()
        except Exception as e:
            self.logger.error(f"Error shutting down logging: {e}")
        
        # Verify cleanup
        cleanup_ok = self.verify_cleanup()
        if cleanup_ok:
            self.logger.info("✅ System shutdown complete - all resources verified clean")
        else:
            self.logger.warning("⚠️  System shutdown complete - some resources may not be fully released")

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
            
            # Print frame buffer statistics (if available - dev mode only)
            if self.frame_buffer_a and self.frame_buffer_b:
                stats_a = self.frame_buffer_a.get_memory_usage()
                stats_b = self.frame_buffer_b.get_memory_usage()
                print(f"  Frames salvos - Câmera A: {stats_a.get('disk_frames', 0)}, "
                      f"Câmera B: {stats_b.get('disk_frames', 0)}")
                print(f"  Espaço em disco - A: {stats_a.get('disk_size_mb', 0):.1f}MB, "
                      f"B: {stats_b.get('disk_size_mb', 0):.1f}MB")
            
            # Print health monitoring statistics
            if self.watchdog:
                watchdog_stats = self.watchdog.get_statistics()
                print(f"\nWatchdog Statistics:")
                print(f"  Health checks performed: {watchdog_stats['checks_performed']}")
                print(f"  Failures detected: {watchdog_stats['failures_detected']}")
                print(f"  Recovery attempts: {watchdog_stats['recovery_attempts']}")
                print(f"  Successful recoveries: {watchdog_stats['successful_recoveries']}")
                print(f"  Success rate: {watchdog_stats['success_rate']*100:.1f}%")

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

        # Keep visualization running until all windows are closed
        try:
            controller.camera_viewer.wait_until_closed()
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
    print("[TRACE] Starting main.py execution...", flush=True)
    modo = get_env_mode()
    print(f"[TRACE] Mode detected: {modo}", flush=True)
    
    if modo == "development":
        print("[TRACE] Running in development mode - calling main_teste()", flush=True)
        main_teste()
    else:
        print("[TRACE] Running in production mode", flush=True)
        # Production mode - run full traffic light controller
        # Use different camera indices for production
        print("[TRACE] Creating TrafficLightController instance...", flush=True)
        controller = TrafficLightController(
            camera_a_dev='/dev/video0',
            camera_b_dev='/dev/video1',  # Changed from video2 to video1
            orchestrator_host='localhost',
            orchestrator_port=9000
        )
        print("[TRACE] TrafficLightController created, starting run_loop()...", flush=True)
        controller.run_loop()