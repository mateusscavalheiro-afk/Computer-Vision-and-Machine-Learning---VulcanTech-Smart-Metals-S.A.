# VulcanTech Smart Metals S.A.
### Monitoramento Preditivo Industrial, OEE em Tempo Real e Visão Computacional 4.0

Solução integrada *End-to-End* de Indústria 4.0 desenvolvida para as Células Flexíveis de Manufatura (CEL-201 a CEL-204) da **VulcanTech Smart Metals S.A.** O sistema realiza inferência de falhas em tempo real, estima impactos de OEE, atua em malha fechada junto ao CLP industrial e inspeciona cordões de solda e zonas termicamente afetadas (HAZ) via visão computacional OpenCV.

---

## Estrutura do Repositório

```text
VulcanTech-Smart-Metals-S.A/
├── data/
│   ├── raw/                 # Dataset bruto coletado dos sensores IoT
│   └── processed/           # Dataset com saneamento, IEM e janelas móveis
├── Ficheiros_IA/            # Artefatos persistidos (modelo Random Forest e Scaler)
├── notebooks/               # Análise exploratória, tratamento e modelagem
│   ├── tratamento.ipynb
│   ├── eda_correlacao.ipynb
│   └── modelagem_treino.ipynb
├── src/                     # Código-fonte da aplicação
│   ├── api.py               # Microsserviço RESTful em FastAPI
│   └── visao_computacional.py # Pipeline OpenCV (Solda e Termografia HAZ)
├── frontend/                # Interface homem-máquina (IHM / Web Dashboard)
│   ├── index.html
│   ├── style.css
│   └── script.js
├── Dockerfile               # Configuração da imagem de container para Edge Computing
├── docker-compose.yml       # Orquestração do serviço e volumes locais
├── requirements.txt         # Dependências do ecossistema Python
└── README.md                # Documentação e guia de execução
```

---

## Pré-requisitos

- **Python**: versão 3.10 ou superior
- **Git**: para clonagem do repositório
- **Docker e Docker Compose** (opcional, para execução conteinerizada)
- Navegador web moderno (Chrome, Edge ou Firefox)

---

## Instalação e Execução Local

### 1. Clonar o Repositório
```bash
git clone https://github.com/mateusscavalheiro-afk/VulcanTech-Smart-Metals-S.A.git
cd VulcanTech-Smart-Metals-S.A
```

### 2. Configurar o Ambiente Virtual (venv)
```bash
# Criar o ambiente virtual
python -m venv .venv

# Ativar no Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Ou no Prompt de Comando (CMD):
.venv\Scripts\activate.bat
# No Linux/macOS:
source .venv/bin/activate
```

### 3. Instalar Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Inicializar a API FastAPI
Execute o servidor Uvicorn com hot-reload ativo:
```bash
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
A API estará operando em `http://127.0.0.1:8000`. A documentação interativa Swagger pode ser acessada em `http://127.0.0.1:8000/docs`.

### 5. Abrir a Interface Web (Dashboard)
Com a API rodando, abra o arquivo `frontend/index.html` diretamente em seu navegador (ou via extensão Live Server no VS Code).

---

## Execução via Docker (Nó Edge)

Para executar o microsserviço isolado em container:

```bash
# Construir e inicializar o container
docker compose up -d --build

# Verificar logs de execução
docker compose logs -f

# Parar os serviços
docker compose down
```

---

## Como Simular e Testar o Projeto

### Aba 1: Telemetria, CLP & OEE
1. No painel **Sensores IoT de Chão de Fábrica**, selecione a célula e a liga metálica processada.
2. Utilize os botões de simulação rápida de cenários:
   - **0 • Operação Nominal**: Mantém parâmetros normais, OEE elevado e comando CLP `MANTER_OPERACAO`.
   - **1 • Desgaste Fuso**: Simula sobretorque e vibração alta, recomendando `REDUZIR_VELOCIDADE_50%`.
   - **2 • Falha Refrigeração**: Simula temperatura crítica e queda de vazão, acionando alívio térmico.
   - **3 • Trinca / Desalinhamento**: Simula pico acústico e perda de pressão, acionando `INTERROMPER_EMERGENCIA`.
3. Clique em **Executar Inferência & Atuar no CLP** para disparar o diagnóstico de engenharia, atualizar os gauges de OEE e registrar o comando no terminal industrial.

### Aba 2: Laboratório de Visão Computacional
1. Alterne entre os modos **Inspeção de Solda** e **Termografia HAZ**.
2. Clique em **Anexar Imagem Real** para carregar fotos de cordão ou mapas térmicos, ou teste com os botões **Amostra Boa** e **Amostra Ruim**.
3. O sistema exibirá a comparação lado a lado (*Imagem Original* vs. *Mapa de Diagnóstico OpenCV*), informando desvios de linearidade, variação transversal de largura, detecção de poros e laudo técnico para o robô de solda.
