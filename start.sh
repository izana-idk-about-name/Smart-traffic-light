#!/bin/bash
# Script de inicialização do sistema de controle de semáforos

# Detectar se está no Raspberry Pi
if [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    echo "🍓 Detectado Raspberry Pi - Usando configurações otimizadas"
    export MODO=rpi
else
    echo "🖥️  Ambiente desktop detectado"
    export MODO=desktop
fi

# Verificar se é modo desenvolvimento
if [ "$1" = "dev" ] || [ "$1" = "development" ]; then
    echo "🔧 Modo desenvolvimento ativado"
    export MODO=development
fi

# Verificar dependências
echo "📦 Verificando dependências..."
python3 -c "import cv2, numpy, sklearn" 2>/dev/null || {
    echo "❌ Dependências não encontradas. Execute: pip install -r requirements.txt"
    exit 1
}

# Executar testes
echo "🧪 Executando testes rápidos..."
python3 test_system.py --quick || {
    echo "⚠️  Alguns testes falharam. Verifique o sistema."
    read -p "Continuar mesmo assim? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
}

# Iniciar sistema
echo "🚀 Iniciando sistema de controle de semáforos..."
python3 main.py