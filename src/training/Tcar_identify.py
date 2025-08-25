import cv2
from src.models.car_identify import CarIdentifier

# Exemplo de teste rápido usando uma imagem da pasta Data
if __name__ == "__main__":
    # Altere o nome do arquivo conforme necessário
    img_path = "src/Data/0410.png"
    img = cv2.imread(img_path)
    if img is None:
        print(f"Não foi possível carregar a imagem: {img_path}")
    else:
        car_identifier = CarIdentifier()
        count = car_identifier.count_cars(img)
        print(f"Carros detectados na imagem: {count}")
        cv2.imshow("Imagem de Teste", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()