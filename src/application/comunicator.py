import socket
import json
import time
import threading
import os
from typing import Optional, Dict, Any

class OrchestratorComunicator:
    """
    Communicator class for sending traffic light decisions and status to orchestrator system.
    Supports both WebSocket and TCP socket communication.
    """
    def __init__(self, host: str = 'localhost', port: int = 9000, use_websocket: bool = False):
        self.host = host
        self.port = port
        self.use_websocket = use_websocket
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self._reconnect_thread = None
        self._reconnect_interval = 5  # seconds
        self._max_retries = 3

        # Initialize connection
        self._connect()

        # Start auto-reconnect thread
        self._start_reconnect_thread()

    def _connect(self) -> bool:
        """Establish connection to orchestrator"""
        try:
            if self.socket:
                self.socket.close()

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"Connected to orchestrator at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to orchestrator: {e}")
            self.connected = False
            return False

    def _start_reconnect_thread(self):
        """Start background thread for auto-reconnection"""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return

        self._reconnect_thread = threading.Thread(
            target=self._auto_reconnect,
            daemon=True
        )
        self._reconnect_thread.start()

    def _auto_reconnect(self):
        """Auto-reconnect mechanism"""
        while True:
            if not self.connected:
                if self._connect():
                    print("Reconnected to orchestrator")
                else:
                    time.sleep(self._reconnect_interval)
            else:
                time.sleep(1)  # Check connection status every second

    def _send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to orchestrator"""
        if not self.connected or not self.socket:
            print("Not connected to orchestrator")
            return False

        try:
            # Convert message to JSON
            json_message = json.dumps(message)
            data = json_message.encode('utf-8')

            # Send message length first (4 bytes)
            length = len(data)
            self.socket.send(length.to_bytes(4, byteorder='big'))

            # Send message data
            self.socket.send(data)

            print(f"Message sent: {message}")
            return True

        except Exception as e:
            print(f"Failed to send message: {e}")
            self.connected = False
            return False

    def send_decision(self, decision: str) -> bool:
        """
        Send traffic light decision to orchestrator

        Args:
            decision: "A" or "B" indicating which direction should have green light

        Returns:
            bool: True if message was sent successfully
        """
        message = {
            "type": "decision",
            "direction": decision,
            "timestamp": time.time()
        }
        return self._send_message(message)

    def send_status(self, count_a: int, count_b: int) -> bool:
        """
        Send current traffic status to orchestrator

        Args:
            count_a: Number of cars detected in direction A
            count_b: Number of cars detected in direction B

        Returns:
            bool: True if message was sent successfully
        """
        message = {
            "type": "status",
            "count_a": count_a,
            "count_b": count_b,
            "timestamp": time.time()
        }
        return self._send_message(message)

    def close(self):
        """Close connection to orchestrator"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        print("Connection to orchestrator closed")