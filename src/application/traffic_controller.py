import cv2
import time
import random
import asyncio
import websockets
from src.models.car_identify import create_car_identifier

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1
ORCHESTRATOR_WS_URL = "ws://localhost:9000"  # ajuste conforme necessário

def count_cars_from_camera(camera_index, identifier):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return 0
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return 0
    return identifier.count_cars(frame)

async def send_decision(decision):
    async with websockets.connect(ORCHESTRATOR_WS_URL) as ws:
        await ws.send(decision)

async def main_loop():
    identifier = create_car_identifier(mode='rpi', use_ml=True, use_custom_model=True)
    print("Sistema de controle de tráfego iniciado (offline, otimizado para Raspberry Pi)")
    while True:
        cars_a = count_cars_from_camera(CAMERA_A_INDEX, identifier)
        cars_b = count_cars_from_camera(CAMERA_B_INDEX, identifier)
        print(f"Carros detectados - A: {cars_a} | B: {cars_b}")

        if cars_a > cars_b:
            decision = "A"
        elif cars_b > cars_a:
            decision = "B"
        else:
            decision = random.choice(["A", "B"])

        print(f"Decisão: liberar {decision}")
        try:
            await send_decision(decision)
            print("Decisão enviada ao orquestrador via websocket.")
        except Exception as e:
            print(f"Erro ao enviar decisão via websocket: {e}")

        time.sleep(10)

if __name__ == "__main__":
    asyncio.run(main_loop())