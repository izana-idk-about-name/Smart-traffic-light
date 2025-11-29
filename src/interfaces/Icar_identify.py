from abc import ABC,abstractmethod

class ICamera_Access(ABC):
    @abstractmethod
    def access_camera():
        """
        este metodo acessa a camera
        """
        ...

class Icar_identify(ABC):
    @abstractmethod
    def Neural_Network():
        """
        este metodo gera a decisao final
        """
        ...