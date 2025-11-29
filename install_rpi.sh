#!/bin/bash
# Script de instalação para Raspberry Pi 4
# Sistema de Controle de Semáforos com IA

set -e  # Parar em caso de erro

echo "========================================="
echo "Instalação do Sistema de Controle de Semáforos"
echo "========================================="
echo ""

# Verificar se está rodando no Raspberry Pi
if ! grep -q "BCM" /proc/cpuinfo; then
    echo "⚠️  Este script é otimizado para Raspberry Pi"
    read -p "Continuar mesmo assim? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

# Atualizar sistema
echo "📦 Atualizando sistema..."
sudo apt update && sudo apt upgrade -y

# Instalar dependências do sistema
echo "🔧 Instalando dependências do sistema..."
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-venv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    libjasper-dev \
    libqtgui4 \
    libqt4-test \
    libqtcore4

# Criar ambiente virtual Python
echo "🐍 Criando ambiente virtual Python..."
python3 -m venv venv
source venv/bin/activate

# Atualizar pip
pip install --upgrade pip

# Instalar dependências Python
echo "📚 Instalando dependências Python..."
pip install -r requirements.txt

# Criar diretórios necessários
echo "📁 Criando estrutura de diretórios..."
mkdir -p logs
mkdir -p data/models
mkdir -p data/training

# Criar arquivo de configuração .env se não existir
if [ ! -f .env ]; then
    echo "⚙️  Criando arquivo de configuração .env..."
    cat > .env << EOF
# Configuração do Sistema de Semáforos Inteligente

# Modo de operação
MODO=production

# Configuração de câmeras
CAMERA_A_INDEX=0
CAMERA_B_INDEX=1

# Configuração de rede
ORCHESTRATOR_HOST=localhost
ORCHESTRATOR_PORT=9000

# Configuração de log
LOG_LEVEL=INFO
LOG_FILE=logs/traffic_light.log

# Configuração de performance
DECISION_INTERVAL=3
MAX_PROCESSING_TIME=1.0
EOF
fi

# Criar script de inicialização
echo "🚀 Criando script de inicialização..."
cat > start.sh << 'EOF'
#!/bin/bash
# Script de inicialização do sistema

echo "Iniciando Sistema de Controle de Semáforos..."
echo "Modo: $(grep MODO .env | cut -d'=' -f2)"
echo ""

# Ativar ambiente virtual
source venv/bin/activate

# Verificar câmeras
echo "Verificando câmeras..."
python3 -c "
import cv2
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'✓ Camera {i}: OK ({frame.shape[1]}x{frame.shape[0]})')
        cap.release()
"

# Iniciar sistema
echo ""
echo "Iniciando processamento..."
python3 main.py
EOF

chmod +x start.sh

# Criar script de teste
echo "🧪 Criando script de teste..."
cat > test.sh << 'EOF'
#!/bin/bash
# Script de teste do sistema

echo "Executando testes do sistema..."
echo ""

# Ativar ambiente virtual
source venv/bin/activate

# Teste de configuração
echo "=== Teste de Configuração ==="
python3 src/settings/rpi_config.py

# Teste de modelo
echo ""
echo "=== Teste de Modelo ==="
python3 -c "
from src.models.car_identify import create_car_identifier
try:
    identifier = create_car_identifier('rpi')
    print('✓ Modelo carregado com sucesso')
except Exception as e:
    print(f'✗ Erro ao carregar modelo: {e}')
"

# Teste de câmeras
echo ""
echo "=== Teste de Câmeras ==="
python3 -c "
import cv2
import time
for i in range(4):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f'✓ Camera {i}: {frame.shape[1]}x{frame.shape[0]} - OK')
            cap.release()
            break
    cap.release()
"

# Teste completo
echo ""
echo "=== Teste Completo ==="
MODO=development python3 main.py
EOF

chmod +x test.sh

# Criar serviço systemd (opcional)
echo "🔧 Criando serviço systemd..."
sudo tee /etc/systemd/system/traffic-light.service > /dev/null << EOF
[Unit]
Description=Sistema de Controle de Semáforos Inteligente
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/traffic-light
ExecStart=/home/pi/traffic-light/venv/bin/python3 /home/pi/traffic-light/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Configurar permissões
sudo systemctl daemon-reload

echo ""
echo "========================================="
echo "Instalação concluída! 🎉"
echo "========================================="
echo ""
echo "Próximos passos:"
echo "1. Conecte as duas câmeras USB"
echo "2. Execute: ./test.sh"
echo "3. Para iniciar o sistema: ./start.sh"
echo "4. Para rodar como serviço: sudo systemctl enable traffic-light.service"
echo ""
echo "Arquivos criados:"
echo "- start.sh    : Script de inicialização"
echo "- test.sh     : Script de teste"
echo "- .env        : Configurações do sistema"
echo "- logs/       : Diretório de logs"
echo ""
echo "Para desinstalar o serviço:"
echo "sudo systemctl disable traffic-light.service"
echo "sudo rm /etc/systemd/system/traffic-light.service"