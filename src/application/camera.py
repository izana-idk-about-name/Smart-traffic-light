import cv2
from typing import Optional
from src.settings.config import ENVIRONMENT
from src.utils.resource_manager import CameraContextManager, ResourceTracker, get_global_tracker
from src.utils.logger import get_logger


class CameraAccess:
    """
    Camera access wrapper with proper resource management.
    
    Features:
    - Automatic resource cleanup
    - Resource leak detection
    - Context manager support
    - Integration with global resource tracker
    """
    
    def __init__(self):
        """Initialize camera access with resource tracking."""
        self.logger = get_logger(__name__)
        self.tracker = get_global_tracker()
    
    def access_camera(self, camera_index: int = 0) -> Optional[bool]:
        """
        Acessa a câmera e exibe o feed em tempo real com gerenciamento de recursos.
        
        Args:
            camera_index: Índice da câmera a ser acessada (padrão: 0)
            
        Returns:
            True se a câmera foi acessada com sucesso, None se houve erro
        """
        window_name = 'Live Webcam Feed'
        
        try:
            # Use context manager for automatic cleanup
            with CameraContextManager(camera_index, self.tracker) as cap:
                if not cap.isOpened():
                    print(f"Não foi possível abrir a câmera {camera_index}")
                    return None
                
                print(f"Câmera {camera_index} acessada com sucesso. Pressione 'q' para sair.")
                
                # Track window
                cv2.namedWindow(window_name)
                self.tracker.track_window(window_name)
                
                while True:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Erro ao capturar frame da câmera")
                        break
                    
                    cv2.imshow(window_name, frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Clean up window
                self.tracker.destroy_window(window_name)
                
                return True
                    
        except Exception as e:
            self.logger.error(f"Erro ao acessar ou processar imagem da câmera: {e}", exc_info=True)
            print(f"Erro ao acessar ou processar imagem da câmera: {e}")
            return None


class ManagedCamera:
    """
    Context manager wrapper for OpenCV VideoCapture with resource tracking.
    
    Usage:
        with ManagedCamera(camera_index=0) as camera:
            ret, frame = camera.read()
            # ... use camera ...
        # Camera automatically released
    """
    
    def __init__(self, camera_index: int = 0, tracker: Optional[ResourceTracker] = None):
        """
        Initialize managed camera.
        
        Args:
            camera_index: Camera device index
            tracker: Optional resource tracker (uses global if None)
        """
        self.camera_index = camera_index
        self.tracker = tracker or get_global_tracker()
        self.camera: Optional[cv2.VideoCapture] = None
        self.logger = get_logger(__name__)
    
    def __enter__(self) -> cv2.VideoCapture:
        """Open camera and return VideoCapture instance."""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
            # Track with resource tracker
            self.tracker.track_camera(self.camera)
            
            self.logger.info(f"Opened camera {self.camera_index}")
            return self.camera
            
        except Exception as e:
            self.logger.error(f"Failed to open camera {self.camera_index}: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release camera resources."""
        if self.camera is not None:
            try:
                self.tracker.release_camera(self.camera)
                self.camera = None
                self.logger.info(f"Released camera {self.camera_index}")
            except Exception as e:
                self.logger.warning(f"Error releasing camera {self.camera_index}: {e}")
        
        return False  # Don't suppress exceptions
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self.camera is not None and self.camera.isOpened()
    
    def read(self):
        """Read frame from camera."""
        if self.camera is None:
            raise RuntimeError("Camera not opened. Use within context manager.")
        return self.camera.read()
    
    def get(self, prop):
        """Get camera property."""
        if self.camera is None:
            raise RuntimeError("Camera not opened. Use within context manager.")
        return self.camera.get(prop)
    
    def set(self, prop, value):
        """Set camera property."""
        if self.camera is None:
            raise RuntimeError("Camera not opened. Use within context manager.")
        return self.camera.set(prop, value)
