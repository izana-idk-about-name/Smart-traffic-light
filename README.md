# Sistema de Controle de Semáforos com IA 🤖

Sistema inteligente para controle de semáforos baseado em **inteligência artificial** e visão computacional, otimizado para Raspberry Pi 4.

## 🎯 Objetivo

Controlar dois semáforos de forma inteligente, analisando o fluxo de veículos em tempo real através de duas câmeras webcams.

## 📋 Funcionalidades

- 🤖 **Detecção de veículos com IA** usando machine learning (MobileNet SSD)
- 🎯 **Precisão superior** comparada à visão computacional tradicional
- ⚡ **Decisão inteligente** baseada no número de veículos em cada direção
- 🖥️ **Otimização para Raspberry Pi 4** com configurações específicas
- 🌐 **Comunicação com orquestrador** via TCP/WebSocket
- 📊 **Monitoramento de performance** e estatísticas em tempo real
- 🔄 **Modo híbrido** com fallback para visão computacional tradicional
- 🧪 **Modo de teste** para desenvolvimento

## 🏗️ Arquitetura

```
src/
├── application/
│   ├── camera.py          # Interface de câmera
│   └── comunicator.py     # Comunicação com orquestrador
├── models/
│   ├── car_identify.py    # 🚗 Modelo de IA para identificação de carros
│   └── download_models.py # 📥 Download de modelos ML pré-treinados
├── settings/
│   ├── config.py          # ⚙️ Configurações gerais
│   └── rpi_config.py      # 🖥️ Configurações específicas para Raspberry Pi
├── training/
│   └── capture_training_data.py # 🎯 Captura de dados para treinamento
├── Data/                  # 📊 Dados de treinamento
├── main.py               # 🎮 Aplicação principal
└── requirements.txt      # 📦 Dependências
setup_ai_system.py       # 🔧 Setup automático com IA
```

## 🚀 Instalação

### 1. Setup Automático com IA (Recomendado) 🚀

```bash
# Setup completo com download de modelos IA
python3 setup_ai_system.py
```

### 2. Preparação do Raspberry Pi Manual

```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências do sistema
sudo apt install python3-pip python3-opencv libatlas-base-dev -y

# Instalar dependências Python
pip3 install -r requirements.txt

# Download de modelos IA pré-treinados
python3 src/models/download_models.py
```

### 2. Configuração das Câmeras

#### Para Raspberry Pi Camera Module v2:
```bash
# Instalar suporte para Pi Camera
sudo apt install python3-picamera2 -y
```

#### Para Webcams USB:
```bash
# Verificar câmeras conectadas
ls /dev/video*
```

### 3. Configuração de Ambiente

```bash
# Copiar arquivo de configuração
cp .env.example .env

# Editar configurações
nano .env
```

## ⚙️ Configuração

### Variáveis de Ambiente (.env)

```bash
# Modo de operação
MODO=production  # ou 'development' para testes

# Configuração de câmeras
CAMERA_A_INDEX=0
CAMERA_B_INDEX=1

# Configuração de rede
ORCHESTRATOR_HOST=localhost
ORCHESTRATOR_PORT=9000

# Configuração de log
LOG_LEVEL=INFO
```

### Configuração para Raspberry Pi

O sistema detecta automaticamente se está rodando em Raspberry Pi e aplica otimizações:

- **Resolução reduzida**: 320x240 pixels
- **FPS otimizado**: 10 fps
- **Processamento otimizado**: Menor uso de CPU e memória
- **Intervalo de decisão**: 3 segundos

## 🤖 Sistema de Inteligência Artificial

### Detecção com IA vs Visão Computacional

| Aspecto | IA (MobileNet SSD) | Visão Computacional (MOG2) |
|---|---|---|
| **Precisão** | ⭐⭐⭐⭐⭐ Alta precisão | ⭐⭐⭐ Boa em condições ideais |
| **Robustez** | ⭐⭐⭐⭐⭐ Funciona bem em condições variadas | ⭐⭐⭐ Sensível a iluminação |
| **Velocidade** | ⭐⭐⭐⭐ Rápida no RPi | ⭐⭐⭐⭐⭐ Muito rápida |
| **Tipo** | Machine Learning | Algoritmo estatístico |
| **Uso** | Detecção precisa de objetos | Motion detection básica |

### Configuração da IA

A IA é habilitada automaticamente se os modelos estiverem disponíveis:

```bash
# Verificar se IA está funcionando
python3 -c "
from src.models.car_identify import create_car_identifier
identifier = create_car_identifier()
print('IA ativa:', identifier.model_loaded)
"
```

### Modelos Utilizados

- **MobileNet SSD**: Modelo pré-treinado no COCO dataset
- **TensorFlow**: Framework de ML para inferência
- **OpenCV DNN**: Interface para execução de modelos

## 🎮 Uso

### Modo Produção (Raspberry Pi)
```bash
python3 main.py
```

### Modo Desenvolvimento/Teste
```bash
# Executar teste básico
MODO=development python3 main.py

# Ou definir no .env
echo "MODO=development" >> .env
python3 main.py
```

### Verificar Configuração
```bash
python3 src/settings/rpi_config.py
```

## 📊 Monitoramento

### Logs de Performance
O sistema exibe estatísticas a cada 10 ciclos:
- Número de veículos em cada direção
- Tempo médio de processamento
- Decisão tomada

### Arquivo de Log
```bash
# Ver logs em tempo real
tail -f traffic_light.log
```

## 🔧 Solução de Problemas

### Câmeras não detectadas
```bash
# Listar dispositivos de vídeo
v4l2-ctl --list-devices

# Testar câmeras individualmente
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Performance lenta
1. Verificar uso de CPU:
   ```bash
   htop
   ```
2. Verificar uso de memória:
   ```bash
   free -h
   ```
3. Reduzir resolução no arquivo `src/settings/rpi_config.py`

### Erros de comunicação
1. Verificar conectividade:
   ```bash
   ping localhost
   ```
2. Verificar porta:
   ```bash
   netstat -tuln | grep 9000
   ```

## 🧪 Testes

### Teste de Câmeras
```bash
python3 -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        print(f'Camera {i}: {ret}, shape: {frame.shape if ret else None}')
    cap.release()
"
```

### Teste de Modelo
```bash
python3 -c "
from src.models.car_identify import create_car_identifier
identifier = create_car_identifier('rpi')
print('Modelo carregado com sucesso')
"
```

## 📈 Otimizações para Raspberry Pi

### 1. Redução de Resolução
- Câmeras: 320x240 (em vez de 640x480)
- Processamento: 4x mais rápido

### 2. Otimização de Memória
- Limite de memória: 512MB
- Garbage collection automático

### 3. Otimização de CPU
- Uso de threads limitado a 2 cores
- Processamento em lote a cada 3 segundos

### 4. Redução de Dependências
- Sem TensorFlow completo (usar TensorFlow Lite se necessário)
- OpenCV otimizado para ARM

## 🔌 Hardware Recomendado

### Raspberry Pi 4
- **Modelo**: 4GB RAM ou superior
- **Armazenamento**: Cartão SD de 32GB classe 10
- **Fonte**: 5V 3A

### Câmeras
- **Opção 1**: 2x Webcams USB 720p (recomendado Logitech C270)
- **Opção 2**: 2x Raspberry Pi Camera Module v2

### Conexões
- **USB**: Portas USB 3.0 para webcams
- **Rede**: Ethernet ou Wi-Fi 2.4GHz/5GHz

## 📋 Checklist de Instalação

- [ ] Raspberry Pi 4 configurado com Raspberry Pi OS
- [ ] Python 3.9+ instalado
- [ ] OpenCV instalado (`sudo apt install python3-opencv`)
- [ ] Dependências Python instaladas (`pip3 install -r requirements.txt`)
- [ ] Câmeras conectadas e testadas
- [ ] Arquivo `.env` configurado
- [ ] Teste básico executado (`MODO=development python3 main.py`)
- [ ] Sistema funcionando em produção

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🆘 Suporte

Para problemas ou dúvidas:
1. Verifique a seção de solução de problemas
2. Abra uma issue no GitHub
3. Consulte os logs em `traffic_light.log`