import socket
import json

class OrchestratorComunicator:
    def __init__(self, host='localhost', port=9000):
        self.host = host
        self.port = port

    def send_decision(self, decision):
        """
        Sends a JSON message with the decision ("A" or "B") to the orchestrator.
        Args:
            decision (str): "A" or "B"
        """
        message = json.dumps({"decision": decision})
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(message.encode('utf-8'))
                # Optionally receive response
                # response = s.recv(1024)
        except Exception as e:
            print(f"Erro ao enviar decisão para o orquestrador: {e}")