# Sistema de Semáforo Inteligente com IA

## Visão Geral

Projeto para controle de dois semáforos usando IA de visão computacional no Raspberry Pi 4. O sistema utiliza duas webcams para monitorar o tráfego e decide qual semáforo abrir com base na quantidade de carros detectados em cada faixa.

---

## Instalação

1. Instale o Python 3.7+ no Raspberry Pi 4.
2. Instale as dependências do projeto:
   ```bash
   pip3 install -r requirements.txt
   ```
3. Conecte duas webcams USB nas portas do Raspberry Pi.

---

## Como Inicializar o Projeto

1. Certifique-se de que o servidor do orquestrador está rodando e configurado para receber conexões na porta definida em `src/application/comunicator.py` (padrão: 9000).
2. Crie um arquivo `.env` na raiz do projeto com o conteúdo abaixo para rodar o modo de teste rápido:
   ```
   ENVIRONMENT=development
   ```
   Para rodar em produção (usando as webcams), basta remover o arquivo `.env` ou definir qualquer valor diferente de `development`.
3. No terminal, execute:
   ```bash
   python3 main.py
   ```
4. O sistema irá capturar imagens das duas câmeras, contar os carros em cada faixa, decidir qual semáforo abrir e enviar a decisão ao orquestrador automaticamente a cada ciclo.
5. Se aparecer a mensagem "Não foi possível abrir a câmera X", verifique se as webcams estão conectadas corretamente e se os índices (0 e 1) correspondem aos dispositivos disponíveis.

---

## Modo de Teste Rápido

- Com `ENVIRONMENT=development` no `.env`, o sistema irá carregar a imagem `src/Data/0410.png`, contar os carros e exibir o resultado na tela.
- Para voltar ao modo normal, remova o `.env` ou altere o valor de `ENVIRONMENT`.

---

## Estrutura do Projeto

- `main.py`: Loop principal integrando câmera, detecção de carros, lógica de decisão e comunicação. Também executa o teste rápido conforme o modo.
- `src/models/car_identify.py`: Lógica de contagem de carros
- `src/application/camera.py`: Utilitário de acesso às câmeras
- `src/application/comunicator.py`: Comunicação com o orquestrador
- `requirements.txt`: Dependências Python
- `src/Data/`: Imagens de treino e teste
- `src/training/`: Scripts de treinamento do modelo
- `passos_projeto.md`: Etapas e documentação do projeto

---

## Observações

- A detecção de carros é básica e otimizada para desempenho, não para precisão.
- Para melhores resultados, garanta iluminação estável e posicionamento fixo das câmeras.
- Toda a lógica foi projetada para fácil adaptação e melhorias futuras.

---

## Licença

MIT License