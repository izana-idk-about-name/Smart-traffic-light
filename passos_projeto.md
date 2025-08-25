# Passos Detalhados do Projeto de Controle de Semáforos com IA

## Etapas já realizadas

- Objetivo definido: sistema para controlar dois semáforos utilizando IA de visão computacional, rodando em Raspberry Pi 4, com duas webcams e dois pontos de semáforo.
- Dados de treinamento organizados na pasta [`src/Data/`] (imagens de carros para identificação).
- Captura de vídeo das duas webcams implementada em [`src/application/camera.py`], utilizando OpenCV.
- Interface para identificação de carros criada em [`src/interfaces/Icar_identify.py`].
- Modelo de identificação de carros implementado em [`src/models/car_identify.py`].
- Interface para controle de semáforo criada em [`src/interfaces/Isemaforo_controller.py`].

## Etapas pendentes

- Implementar lógica de decisão no código principal para:
  - Receber imagens das duas webcams.
  - Contar o número de carros em cada faixa usando o modelo de [`src/models/car_identify.py`].
  - Decidir qual semáforo (A ou B) deve abrir para diminuir a quantidade de carros.
  - Gerar e enviar sinal JSON (ou via WebSocket) com mensagem "A" ou "B" para o orquestrador.
- Implementar comunicação entre a IA e o orquestrador em [`src/application/comunicator.py`], suportando JSON e, se necessário, WebSocket para leveza.
- Integrar o controle dos dois semáforos e das duas webcams no fluxo principal em [`main.py`], garantindo que os sinais sejam enviados corretamente e em tempo real.
- Adaptar todo o pipeline para rodar eficientemente no Raspberry Pi 4:
  - Garantir uso de OpenCV para processamento de vídeo.
  - Otimizar código para baixo consumo de recursos.
  - Eliminar dependências pesadas como PyTorch, se possível.
- Testar o sistema completo no hardware alvo (Raspberry Pi 4), validando funcionamento com as webcams e semáforos reais.
- Especificar todas as dependências e requisitos no arquivo [`requirements.txt`] para facilitar a instalação no Raspberry Pi 4.
