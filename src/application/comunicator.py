import socket
import json
import threading
import time
import os
from typing import Optional

class OrchestratorComunicator:
    def __init__(self, host='localhost', port=9000, use_websocket=False):
        """
        Initialize the orchestrator communicator

        Args:
            host: Host address of the orchestrator
            port: Port number for communication
            use_websocket: Legacy parameter, currently unused (always uses TCP socket)
        """
        self.host = host
        self.port = port
        self.use_websocket = use_websocket
        self.socket = None
        self.connected = False
        
    def _create_connection(self) -> bool:
        """Create socket connection to orchestrator"""
        try:
            # Use TCP socket (WebSocket implementation would require websockets library)
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))

            self.connected = True
            return True
        except Exception as e:
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: Erro ao conectar ao orquestrador: {e}")
            self.connected = False
            return False
    
    def _close_connection(self):
        """Close socket connection"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            finally:
                self.socket = None
                self.connected = False
    
    def send_decision(self, decision: str) -> bool:
        """
        Send decision to orchestrator

        Args:
            decision: "A" or "B" indicating which traffic light to open

        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        if decision not in ["A", "B"]:
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: Decisão inválida: {decision}. Use 'A' ou 'B'.")
            return False

        # In development mode, skip network communication
        if os.getenv('MODO', '').lower() == 'development':
            print(f"Debug: Enviando decisão simulada: {decision}")
            return True
            
        message = json.dumps({
            "decision": decision,
            "timestamp": time.time(),
            "type": "traffic_light_control"
        })
        
        try:
            # Try to create connection if not already connected
            if not self.connected:
                if not self._create_connection():
                    return False
            
            # Send message
            self.socket.sendall(message.encode('utf-8'))
            
            # Optional: Wait for acknowledgment
            try:
                response = self.socket.recv(1024)
                if response:
                    ack = json.loads(response.decode('utf-8'))
                    if ack.get('status') == 'ok':
                        return True
            except socket.timeout:
                # No acknowledgment received, but message was sent
                pass
            except:
                pass
                
            return True
            
        except Exception as e:
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: Erro ao enviar decisão: {e}")
            self._close_connection()
            return False
    
    def send_status(self, camera_a_count: int, camera_b_count: int) -> bool:
        """
        Send status update with car counts
        
        Args:
            camera_a_count: Number of cars detected in camera A
            camera_b_count: Number of cars detected in camera B
            
        Returns:
            bool: True if message was sent successfully
        """
        message = json.dumps({
            "type": "status_update",
            "camera_a": camera_a_count,
            "camera_b": camera_b_count,
            "timestamp": time.time()
        })
        
        # In development mode, skip network communication
        if os.getenv('MODO', '').lower() == 'development':
            print(f"Debug: Enviando status simulado: A={camera_a_count}, B={camera_b_count}")
            return True

        try:
            if not self.connected:
                if not self._create_connection():
                    return False

            self.socket.sendall(message.encode('utf-8'))
            return True

        except Exception as e:
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: Erro ao enviar status: {e}")
            self._close_connection()
            return False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self._close_connection()

class MockOrchestrator:
    """
    Mock orchestrator for testing purposes
    Can be used to simulate the orchestrator during development
    """
    def __init__(self, host='localhost', port=9000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
    def start(self):
        """Start mock orchestrator server"""
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)
        
        print(f"Mock orchestrator iniciado em {self.host}:{self.port}")
        
        def accept_connections():
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"Conexão recebida de {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Erro no servidor: {e}")
        
        server_thread = threading.Thread(target=accept_connections)
        server_thread.daemon = True
        server_thread.start()
        
    def _handle_client(self, client_socket):
        """Handle individual client connection"""
        try:
            while True:
                data = client_socket.recv(1024)
                if not data:
                    break
                
                try:
                    message = json.loads(data.decode('utf-8'))
                    print(f"Mensagem recebida: {message}")
                    
                    # Send acknowledgment
                    response = json.dumps({"status": "ok"})
                    client_socket.sendall(response.encode('utf-8'))
                    
                except json.JSONDecodeError:
                    print("Erro ao decodificar JSON")
                    
        except Exception as e:
            print(f"Erro ao lidar com cliente: {e}")
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop mock orchestrator"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

if __name__ == "__main__":
    # Run mock orchestrator for testing
    mock = MockOrchestrator()
    mock.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        mock.stop()
        print("Mock orchestrador encerrado")