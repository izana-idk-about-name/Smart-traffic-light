import cv2
from typing import Optional
from src.settings.config import ENVIRONTMENT
class CameraAccess:
    def access_camera(self, camera_index: int = 0) -> Optional[bool]:
        """
        Acessa a câmera e exibe o feed em tempo real.
        
        Args:
            camera_index: Índice da câmera a ser acessada (padrão: 0)
            
        Returns:
            True se a câmera foi acessada com sucesso, None se houve erro
        """
        env = [0 if ENVIRONTMENT == "development" else "Data/images.jpeg"]
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"Não foi possível abrir a câmera {camera_index}")
                return None
            
            print(f"Câmera {camera_index} acessada com sucesso. Pressione 'q' para sair.")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Erro ao capturar frame da câmera")
                    break
                
                cv2.imshow('Live Webcam Feed', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Erro ao acessar ou processar imagem da câmera: {e}")
            return None
            
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            
        return True
    
app = CameraAccess()
app.access_camera()