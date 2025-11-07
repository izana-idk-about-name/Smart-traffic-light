# Guia de Produção - Smart Traffic Light

## 📋 Visão Geral

Este guia fornece um checklist completo e instruções detalhadas para deployment do sistema Smart Traffic Light em ambiente de produção, garantindo operação confiável 24/7.

**Versão:** 2.0.0  
**Status:** Production Ready  
**Última Atualização:** 2025-11-07

---

## 🎯 Pré-requisitos de Hardware

### Raspberry Pi 4 (Recomendado)

| Componente | Especificação Mínima | Recomendado |
|------------|---------------------|-------------|
| **Modelo** | Raspberry Pi 4 Model B | Raspberry Pi 4 Model B |
| **RAM** | 4GB | 8GB |
| **Armazenamento** | 32GB Classe 10 | 64GB+ Classe 10/A1 |
| **Fonte** | 5V 3A USB-C | 5V 3A Oficial |
| **Câmeras** | 2x Webcam USB 720p | 2x Webcam USB 1080p |
| **Rede** | Ethernet 100Mbps | Ethernet 1Gbps |
| **Cooling** | Passivo | Ativo (ventilador) |

### Câmeras

**Opção 1: Webcams USB** (Recomendado)
- Logitech C270 ou superior
- Resolução mínima: 720p (1280x720)
- FPS mínimo: 30fps
- Auto-focus desejável

**Opção 2: Raspberry Pi Camera Module**
- Pi Camera Module v2 ou v3
- Requer adaptador/multiplexer para 2 câmeras
- Melhor performance, menor latência

### Rede

- Conexão estável (Ethernet preferível)
- Latência < 50ms para orquestrador
- Largura de banda mínima: 1Mbps

---

## 🚀 Checklist de Deployment

### Fase 1: Preparação do Sistema

#### 1.1 Sistema Operacional
```bash
# ✅ Verificar versão do OS
cat /etc/os-release

# ✅ Atualizar sistema
sudo apt update && sudo apt upgrade -y

# ✅ Instalar dependências do sistema
sudo apt install -y \
    python3.9 \
    python3-pip \
    python3-opencv \
    libatlas-base-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libharfbuzz0b \
    libwebp6 \
    libjasper1 \
    libilmbase23 \
    libopenexr23 \
    libgstreamer1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    git \
    v4l-utils
```

**Checklist:**
- [ ] OS atualizado para versão mais recente
- [ ] Python 3.9+ instalado
- [ ] Todas as dependências do sistema instaladas
- [ ] Git configurado

#### 1.2 Usuário e Permissões
```bash
# ✅ Criar usuário dedicado (opcional, mas recomendado)
sudo adduser trafficlight --disabled-password

# ✅ Adicionar ao grupo video (acesso às câmeras)
sudo usermod -a -G video trafficlight

# ✅ Configurar sudo sem senha para restart
echo "trafficlight ALL=(ALL) NOPASSWD: /bin/systemctl restart trafficlight" | \
    sudo tee /etc/sudoers.d/trafficlight
```

**Checklist:**
- [ ] Usuário dedicado criado
- [ ] Permissões de câmera configuradas
- [ ] Sudo configurado (se necessário)

### Fase 2: Instalação da Aplicação

#### 2.1 Clone e Setup
```bash
# ✅ Clone do repositório
cd /opt
sudo git clone https://github.com/seu-usuario/Smart-traffic-light.git
sudo chown -R trafficlight:trafficlight Smart-traffic-light
cd Smart-traffic-light

# ✅ Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# ✅ Instalar dependências Python
pip install --upgrade pip
pip install -r requirements.txt

# ✅ Download de modelos de IA
python3 src/models/download_models.py
```

**Checklist:**
- [ ] Repositório clonado em `/opt/Smart-traffic-light`
- [ ] Ambiente virtual criado e ativado
- [ ] Todas as dependências Python instaladas
- [ ] Modelos de IA baixados com sucesso

#### 2.2 Configuração
```bash
# ✅ Copiar arquivo de configuração
cp .env.example .env

# ✅ Editar configurações
nano .env
```

**Configuração Produção (.env):**
```bash
# ==========================================
# PRODUÇÃO - CONFIGURAÇÃO SMART TRAFFIC LIGHT
# ==========================================

# Modo de operação
MODO=production

# Câmeras - ajustar índices conforme hardware
CAMERA_A_INDEX=0
CAMERA_B_INDEX=1
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=10
USE_TEST_IMAGES=false

# Rede - ajustar para seu orquestrador
ORCHESTRATOR_HOST=192.168.1.100
ORCHESTRATOR_PORT=9000
USE_WEBSOCKET=true

# Logging
LOG_LEVEL=INFO
LOG_DIR=/var/log/trafficlight

# Performance
MEMORY_LIMIT_MB=512
MAX_FRAMES_SAVED=100
FRAME_SAVE_INTERVAL=100

# Validação de treinamento
MIN_SAMPLES_PER_CLASS=100
VALIDATE_BEFORE_TRAINING=true
```

**Checklist:**
- [ ] Arquivo `.env` criado e configurado
- [ ] Índices de câmera verificados
- [ ] Host e porta do orquestrador corretos
- [ ] Diretório de logs criado

#### 2.3 Validação Pré-Deployment
```bash
# ✅ Testar configurações
python3 test_settings.py

# ✅ Testar logging
python3 test_logger.py

# ✅ Testar câmeras
python3 test_camera_source.py

# ✅ Validar dados de treinamento (se aplicável)
python3 scripts/validate_training_data.py \
    --dataset data \
    --strict \
    --output validation_report.json

# ✅ Teste de integração
MODO=development python3 main.py
# Deixar rodar por 5 minutos, verificar logs
```

**Checklist:**
- [ ] Todas as configurações validadas
- [ ] Sistema de logging funcionando
- [ ] Ambas as câmeras detectadas e funcionais
- [ ] Teste de integração bem-sucedido
- [ ] Sem erros nos logs

### Fase 3: Configuração de Serviço Systemd

#### 3.1 Criar Service Unit
```bash
# ✅ Criar arquivo de serviço
sudo nano /etc/systemd/system/trafficlight.service
```

**Conteúdo do arquivo:**
```ini
[Unit]
Description=Smart Traffic Light Control System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=trafficlight
Group=trafficlight
WorkingDirectory=/opt/Smart-traffic-light
Environment="PATH=/opt/Smart-traffic-light/venv/bin"
ExecStart=/opt/Smart-traffic-light/venv/bin/python3 /opt/Smart-traffic-light/main.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/trafficlight/systemd-stdout.log
StandardError=append:/var/log/trafficlight/systemd-stderr.log

# Limites de recursos
MemoryLimit=1G
CPUQuota=200%

# Segurança
PrivateTmp=yes
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/trafficlight /opt/Smart-traffic-light/logs /opt/Smart-traffic-light/detection_frames

[Install]
WantedBy=multi-user.target
```

#### 3.2 Habilitar e Iniciar Serviço
```bash
# ✅ Recarregar systemd
sudo systemctl daemon-reload

# ✅ Habilitar serviço (iniciar no boot)
sudo systemctl enable trafficlight

# ✅ Iniciar serviço
sudo systemctl start trafficlight

# ✅ Verificar status
sudo systemctl status trafficlight

# ✅ Ver logs em tempo real
sudo journalctl -u trafficlight -f
```

**Checklist:**
- [ ] Service unit criado
- [ ] Serviço habilitado para iniciar no boot
- [ ] Serviço iniciado com sucesso
- [ ] Logs indicam operação normal

### Fase 4: Monitoramento e Observabilidade

#### 4.1 Configurar Rotação de Logs
```bash
# ✅ Criar configuração logrotate
sudo nano /etc/logrotate.d/trafficlight
```

**Conteúdo:**
```
/var/log/trafficlight/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 trafficlight trafficlight
    sharedscripts
    postrotate
        systemctl reload trafficlight > /dev/null 2>&1 || true
    endscript
}
```

#### 4.2 Configurar Monitoramento de Saúde
```bash
# ✅ Criar script de health check
sudo nano /usr/local/bin/trafficlight-healthcheck.sh
```

**Script:**
```bash
#!/bin/bash

LOG_FILE="/var/log/trafficlight/traffic_light.log"
MAX_AGE=300  # 5 minutos

# Verificar se processo está rodando
if ! systemctl is-active --quiet trafficlight; then
    echo "CRITICAL: Serviço não está rodando"
    exit 2
fi

# Verificar idade do último log
if [ -f "$LOG_FILE" ]; then
    LAST_MOD=$(stat -c %Y "$LOG_FILE")
    NOW=$(date +%s)
    AGE=$((NOW - LAST_MOD))
    
    if [ $AGE -gt $MAX_AGE ]; then
        echo "WARNING: Nenhum log novo por ${AGE}s"
        exit 1
    fi
fi

# Verificar erros recentes
ERRORS=$(tail -100 "$LOG_FILE" | grep -c "ERROR\|CRITICAL")
if [ $ERRORS -gt 5 ]; then
    echo "WARNING: $ERRORS erros encontrados nos últimos 100 logs"
    exit 1
fi

echo "OK: Sistema operando normalmente"
exit 0
```

```bash
# ✅ Tornar executável
sudo chmod +x /usr/local/bin/trafficlight-healthcheck.sh

# ✅ Adicionar ao cron (executar a cada 5 minutos)
echo "*/5 * * * * /usr/local/bin/trafficlight-healthcheck.sh >> /var/log/trafficlight/healthcheck.log 2>&1" | sudo crontab -u trafficlight -
```

#### 4.3 Alertas (Opcional)
```bash
# ✅ Instalar ferramentas de alerta
sudo apt install -y mailutils

# ✅ Configurar script de alerta
sudo nano /usr/local/bin/trafficlight-alert.sh
```

**Script de Alerta:**
```bash
#!/bin/bash

EMAIL="admin@example.com"
SUBJECT="[ALERTA] Smart Traffic Light"

# Executar health check
/usr/local/bin/trafficlight-healthcheck.sh
STATUS=$?

if [ $STATUS -ne 0 ]; then
    # Coletar informações
    HOSTNAME=$(hostname)
    UPTIME=$(uptime)
    LAST_LOGS=$(tail -50 /var/log/trafficlight/traffic_light.log)
    
    # Enviar email
    {
        echo "Sistema: $HOSTNAME"
        echo "Uptime: $UPTIME"
        echo ""
        echo "Últimos Logs:"
        echo "$LAST_LOGS"
    } | mail -s "$SUBJECT - Status $STATUS" "$EMAIL"
    
    # Tentar restart se crítico
    if [ $STATUS -eq 2 ]; then
        sudo systemctl restart trafficlight
        echo "Restart automático executado" | mail -s "$SUBJECT - Auto-Recovery" "$EMAIL"
    fi
fi
```

**Checklist:**
- [ ] Rotação de logs configurada
- [ ] Script de health check criado e testado
- [ ] Cron job configurado
- [ ] Alertas configurados (se aplicável)

### Fase 5: Otimizações de Produção

#### 5.1 Performance do Sistema
```bash
# ✅ Configurar limites de recursos
sudo nano /etc/security/limits.conf
```

Adicionar:
```
trafficlight soft nofile 4096
trafficlight hard nofile 8192
trafficlight soft nproc 2048
trafficlight hard nproc 4096
```

#### 5.2 Swap (se necessário)
```bash
# ✅ Criar arquivo de swap (2GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# ✅ Tornar permanente
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# ✅ Ajustar swappiness para SSD
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### 5.3 Otimizações Raspberry Pi
```bash
# ✅ Configurar GPU memory split
sudo raspi-config
# Performance Options -> GPU Memory -> 128MB

# ✅ Habilitar overclock (opcional, com cooling)
# Performance Options -> Overclock -> Modest

# ✅ Desabilitar serviços desnecessários
sudo systemctl disable bluetooth
sudo systemctl disable avahi-daemon
```

**Checklist:**
- [ ] Limites de recursos configurados
- [ ] Swap configurado (se RAM < 8GB)
- [ ] GPU memory alocada adequadamente
- [ ] Serviços desnecessários desabilitados

### Fase 6: Segurança

#### 6.1 Firewall
```bash
# ✅ Instalar UFW
sudo apt install -y ufw

# ✅ Configurar regras básicas
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow from 192.168.1.0/24 to any port 9000  # Orquestrador
sudo ufw enable
```

#### 6.2 Atualizações Automáticas
```bash
# ✅ Instalar unattended-upgrades
sudo apt install -y unattended-upgrades

# ✅ Configurar
sudo dpkg-reconfigure -plow unattended-upgrades
```

#### 6.3 Backup
```bash
# ✅ Criar script de backup
sudo nano /usr/local/bin/trafficlight-backup.sh
```

**Script:**
```bash
#!/bin/bash

BACKUP_DIR="/backup/trafficlight"
DATE=$(date +%Y%m%d-%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup de configurações e logs
tar -czf "$BACKUP_DIR/config-$DATE.tar.gz" \
    /opt/Smart-traffic-light/.env \
    /opt/Smart-traffic-light/logs/ \
    /var/log/trafficlight/

# Manter apenas últimos 7 dias
find "$BACKUP_DIR" -name "config-*.tar.gz" -mtime +7 -delete

echo "Backup concluído: config-$DATE.tar.gz"
```

```bash
# ✅ Tornar executável
sudo chmod +x /usr/local/bin/trafficlight-backup.sh

# ✅ Agendar backup diário
echo "0 2 * * * /usr/local/bin/trafficlight-backup.sh >> /var/log/trafficlight/backup.log 2>&1" | sudo crontab -u root -
```

**Checklist:**
- [ ] Firewall configurado e ativo
- [ ] Atualizações automáticas habilitadas
- [ ] Script de backup criado e agendado
- [ ] Testado restore de backup

---

## 📊 Métricas e KPIs

### Métricas de Sistema

| Métrica | Threshold Normal | Alerta | Crítico |
|---------|------------------|--------|---------|
| CPU Usage | < 60% | > 80% | > 95% |
| Memory Usage | < 70% | > 85% | > 95% |
| Disk Usage | < 70% | > 85% | > 95% |
| Temperature | < 60°C | > 70°C | > 80°C |

### Métricas de Aplicação

| Métrica | Esperado | Investigar se |
|---------|----------|---------------|
| Processing Time | < 0.3s | > 1.0s |
| Detection Accuracy | > 90% | < 80% |
| Recovery Success | > 95% | < 85% |
| Uptime | > 99.5% | < 99% |

### Comandos de Monitoramento

```bash
# CPU e Memória
htop

# Temperatura
vcgencmd measure_temp

# Uso de disco
df -h

# Status do serviço
systemctl status trafficlight

# Logs em tempo real
tail -f /var/log/trafficlight/traffic_light.log

# Performance da aplicação
tail -f /var/log/trafficlight/performance.log

# Estatísticas de health checks
grep "Health\|Watchdog" /var/log/trafficlight/traffic_light.log | tail -50
```

---

## 🔧 Manutenção

### Manutenção Diária

```bash
# ✅ Verificar status
sudo systemctl status trafficlight

# ✅ Verificar logs por erros
grep -i "error\|critical" /var/log/trafficlight/traffic_light.log | tail -20

# ✅ Verificar uso de recursos
htop
df -h
```

### Manutenção Semanal

```bash
# ✅ Verificar estatísticas
grep "Statistics\|Watchdog Stats" /var/log/trafficlight/traffic_light.log | tail -50

# ✅ Verificar health checks
grep "System Health" /var/log/trafficlight/traffic_light.log | tail -30

# ✅ Limpar frames antigos (se necessário)
find /opt/Smart-traffic-light/detection_frames -type f -mtime +7 -delete
```

### Manutenção Mensal

```bash
# ✅ Atualizar sistema
sudo apt update && sudo apt upgrade -y

# ✅ Atualizar dependências Python (cuidado!)
cd /opt/Smart-traffic-light
source venv/bin/activate
pip list --outdated

# ✅ Verificar espaço em disco
du -sh /var/log/trafficlight/*
du -sh /opt/Smart-traffic-light/logs/*

# ✅ Testar backup e restore
/usr/local/bin/trafficlight-backup.sh
```

---

## 🚨 Troubleshooting em Produção

### Sistema não inicia

```bash
# 1. Verificar status do serviço
sudo systemctl status trafficlight

# 2. Ver logs de erro
sudo journalctl -u trafficlight -n 100

# 3. Verificar permissões
ls -la /opt/Smart-traffic-light/

# 4. Verificar configuração
python3 /opt/Smart-traffic-light/test_settings.py

# 5. Tentar iniciar manualmente
cd /opt/Smart-traffic-light
source venv/bin/activate
MODO=development python3 main.py
```

### Performance degradada

```bash
# 1. Verificar recursos
htop
free -h
df -h

# 2. Verificar temperatura
vcgencmd measure_temp

# 3. Verificar tempo de processamento
tail -f /var/log/trafficlight/performance.log

# 4. Verificar health checks
grep "processing_time" /var/log/trafficlight/traffic_light.log | tail -20
```

### Câmeras não funcionando

```bash
# 1. Listar câmeras disponíveis
v4l2-ctl --list-devices

# 2. Testar câmera diretamente
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# 3. Verificar permissões
groups trafficlight | grep video

# 4. Reiniciar USB
sudo usb-devices
# Identificar câmeras e fazer reset se necessário
```

### Alto uso de memória

```bash
# 1. Verificar processos
ps aux | grep python

# 2. Forçar garbage collection via restart
sudo systemctl restart trafficlight

# 3. Verificar configurações
grep "MEMORY_LIMIT\|MAX_FRAMES" /opt/Smart-traffic-light/.env

# 4. Ajustar limites se necessário
nano /opt/Smart-traffic-light/.env
# Reduzir MAX_FRAMES_SAVED
```

---

## 📞 Contatos de Emergência

### Procedimento de Escalação

1. **Nível 1** - Restart automático via watchdog
2. **Nível 2** - Health check falha → Alerta para equipe
3. **Nível 3** - Falha crítica → Escalação imediata

### Comandos de Emergência

```bash
# Restart rápido
sudo systemctl restart trafficlight

# Parar sistema
sudo systemctl stop trafficlight

# Ver últimos 200 logs
sudo journalctl -u trafficlight -n 200 --no-pager

# Backup de emergência
sudo tar -czf /tmp/emergency-backup-$(date +%s).tar.gz \
    /opt/Smart-traffic-light/.env \
    /var/log/trafficlight/ \
    /opt/Smart-traffic-light/logs/
```

---

## ✅ Checklist Final de Produção

### Pré-Deploy
- [ ] Todos os testes passando
- [ ] Documentação revisada
- [ ] Configurações de produção validadas
- [ ] Backup do sistema atual (se upgrade)

### Deploy
- [ ] Sistema instalado conforme guia
- [ ] Serviço systemd configurado
- [ ] Logs funcionando corretamente
- [ ] Monitoramento ativo
- [ ] Alertas configurados

### Pós-Deploy
- [ ] Sistema rodando estável por 24h
- [ ] Health checks todos verdes
- [ ] Performance dentro do esperado
- [ ] Backup automático funcionando
- [ ] Documentação de produção atualizada

### Validação Final
- [ ] Uptime > 99% após 7 dias
- [ ] Zero vazamentos de memória observados
- [ ] Auto-recovery testado e funcionando
- [ ] Alertas recebidos e acionados corretamente
- [ ] Equipe treinada em manutenção e troubleshooting

---

## 📚 Referências

- [RESUMO_CORRECOES.md](../RESUMO_CORRECOES.md) - Histórico de melhorias
- [ARQUITETURA.md](ARQUITETURA.md) - Arquitetura do sistema
- [RESOURCE_MANAGEMENT.md](RESOURCE_MANAGEMENT.md) - Gerenciamento de recursos
- [HEALTH_MONITORING.md](HEALTH_MONITORING.md) - Sistema de monitoramento
- [DATA_VALIDATION.md](DATA_VALIDATION.md) - Validação de dados

---

**Versão do Guia:** 2.0.0  
**Última Revisão:** 2025-11-07  
**Status:** ✅ Production Ready