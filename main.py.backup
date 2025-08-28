import cv2
import os
from src.models.car_identify import CarIdentifier
from src.application.comunicator import OrchestratorComunicator

def get_camera_frame(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Não foi possível abrir a câmera {camera_index}. Verifique se ela está conectada e o índice está correto.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Não foi possível capturar frame da câmera {camera_index}.")
        return None
    return frame

def main_loop():
    car_identifier = CarIdentifier()
    comunicator = OrchestratorComunicator(host='localhost', port=9000)  # Ajuste host/port conforme necessário
    while True:
        frame_a = get_camera_frame(0)
        frame_b = get_camera_frame(1)
        if frame_a is None or frame_b is None:
            print("Erro ao capturar imagens das câmeras.")
            print("Se você está tentando rodar o modo de teste rápido, defina a variável de ambiente MODO=teste antes de executar o main.")
            break

        count_a = car_identifier.count_cars(frame_a)
        count_b = car_identifier.count_cars(frame_b)

        print(f"Carros na faixa A: {count_a} | Carros na faixa B: {count_b}")

        if count_a > count_b:
            decision = "A"
        else:
            decision = "B"

        print(f"Abrir semáforo: {decision}")

        # Envia decisão para o orquestrador
        comunicator.send_decision(decision)

        # TODO: Integrar controle real dos semáforos

        # Aguarda 5 segundos antes de nova decisão
        if cv2.waitKey(5000) & 0xFF == ord('q'):
            break

def main_teste():
    # Teste rápido usando imagem de exemplo
    img_path = "src/Data/0410.png"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Não foi possível carregar a imagem de teste: {img_path}")
        return
    car_identifier = CarIdentifier()
    count = car_identifier.count_cars(img)
    print(f"Carros detectados na imagem de teste: {count}")
    cv2.imshow("Imagem de Teste", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_env_mode():
    # Lê o modo do .env (ENVIRONMENT). Se for 'development', executa teste rápido, senão produção.
    env_path = ".env"
    mode = "production"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip().startswith("ENVIRONMENT"):
                    key, value = line.strip().split("=", 1)
                    mode = value.strip().lower()
    return mode

if __name__ == "__main__":
    modo = get_env_mode()
    if modo == "development":
        main_teste()
    else:
        main_loop()