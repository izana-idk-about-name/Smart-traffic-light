from abc import ABC, abstractmethod

class Isemaforo_controller(ABC):
    @abstractmethod
    def semaforo_decision(decision) -> dict:
        """
        este metodo trasforma a decisao final em um json para o rasberry
        """
        ...