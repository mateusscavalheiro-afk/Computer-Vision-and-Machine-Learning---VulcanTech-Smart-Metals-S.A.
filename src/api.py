import os
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import cv2  # Necessário para decodificar a imagem enviada via UploadFile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importação das funções de visão computacional (OpenCV)
try:
    from src.visao_computacional import inspecionar_cordao_solda, analisar_termografia_haz
except ImportError:
    from visao_computacional import inspecionar_cordao_solda, analisar_termografia_haz

# Inicialização do FastAPI
app = FastAPI(
    title="VulcanTech Smart Metals S.A. - Telemetria e IA Preditiva",
    description="API RESTful integrada com Machine Learning, OEE e Visão Computacional.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Localização flexível do modelo e do scaler
PASTAS_POSSIVEIS = [
    os.path.join(".", "Ficheiros_IA"),
    os.path.join("..", "Ficheiros_IA"),
    os.path.join("C:\\Users\\mateus_s_cavalheiro\\Documents\\GitHub\\VulcanTech-Smart-Metals-S.A", "Ficheiros_IA")
]

modelo = None
scaler = None

for pasta in PASTAS_POSSIVEIS:
    caminho_mod = os.path.join(pasta, "modelo_falhas_vulcantech.joblib")
    caminho_sca = os.path.join(pasta, "scaler_vulcantech.joblib")
    if os.path.exists(caminho_mod) and os.path.exists(caminho_sca):
        modelo = joblib.load(caminho_mod)
        scaler = joblib.load(caminho_sca)
        print(f"[INFO] Artefatos de IA carregados com sucesso de: {pasta}")
        break

# Mapeamento de diagnósticos e ações industriais para CLP
MAPA_DIAGNOSTICOS = {
    0: "0 - Operação Normal",
    1: "1 - Desgaste Severo de Ferramenta / Fuso",
    2: "2 - Obstrução no Sistema de Refrigeração / Superaquecimento",
    3: "3 - Desalinhamento da Matriz / Trinca Mecânica"
}

MAPA_ACOES = {
    0: "MANTER_OPERACAO",
    1: "REDUZIR_VELOCIDADE_50%",
    2: "REDUZIR_VELOCIDADE_50%",
    3: "INTERROMPER_EMERGENCIA"
}

FEATURES_ORDEM = [
    'temperatura_fuso_c', 'vibracao_triaxial_rms_mms', 'torque_fuso_nm',
    'vazao_refrigerante_lmin', 'corrente_estator_amp', 'pressao_hidraulica_bar',
    'nivel_ruido_db', 'iem',
    'temperatura_fuso_c_rolling_mean_30min', 'temperatura_fuso_c_rolling_std_30min',
    'temperatura_fuso_c_rolling_mean_2h', 'temperatura_fuso_c_rolling_std_2h',
    'vibracao_triaxial_rms_mms_rolling_mean_30min', 'vibracao_triaxial_rms_mms_rolling_std_30min',
    'vibracao_triaxial_rms_mms_rolling_mean_2h', 'vibracao_triaxial_rms_mms_rolling_std_2h',
    'torque_fuso_nm_rolling_mean_30min', 'torque_fuso_nm_rolling_std_30min',
    'torque_fuso_nm_rolling_mean_2h', 'torque_fuso_nm_rolling_std_2h'
]

def calcular_oee(classe_predita: int):
    """Calcula o OEE projetado conforme o modo de falha."""
    if classe_predita == 0:
        disp, perf, qual = 0.95, 0.93, 0.98
    elif classe_predita == 1:
        disp, perf, qual = 0.75, 0.85, 0.90
    elif classe_predita == 2:
        disp, perf, qual = 0.80, 0.70, 0.82
    else:
        disp, perf, qual = 0.40, 0.60, 0.70
    return {
        "disponibilidade_pct": round(disp * 100, 2),
        "performance_pct": round(perf * 100, 2),
        "qualidade_pct": round(qual * 100, 2),
        "oee_total_pct": round(disp * perf * qual * 100, 2)
    }

# Modelos Pydantic
class LeituraSensor(BaseModel):
    id_celula: str = Field(default="CEL-201", description="Identificador da célula de manufatura (CEL-201 a CEL-204)")
    temperatura_fuso_c: float = Field(..., description="Temperatura no fuso CNC / matriz (°C)")
    vibracao_triaxial_rms_mms: float = Field(..., description="Vibração triaxial do cabeçote (mm/s RMS)")
    torque_fuso_nm: float = Field(..., description="Torque instantâneo no fuso (Nm)")
    vazao_refrigerante_lmin: float = Field(..., description="Vazão de fluido de corte e refrigeração (L/min)")
    corrente_estator_amp: float = Field(..., description="Corrente do motor elétrico de acionamento (A)")
    pressao_hidraulica_bar: float = Field(..., description="Pressão hidráulica/pneumática da prensa (bar)")
    nivel_ruido_db: float = Field(..., description="Emissão acústica do corte (dB)")
    
    # Features dinâmicas opcionais (caso o sensor envie apenas as 7 grandezas instantâneas)
    iem: Optional[float] = None
    temperatura_fuso_c_rolling_mean_30min: Optional[float] = None
    temperatura_fuso_c_rolling_std_30min: Optional[float] = 0.0
    temperatura_fuso_c_rolling_mean_2h: Optional[float] = None
    temperatura_fuso_c_rolling_std_2h: Optional[float] = 0.0
    vibracao_triaxial_rms_mms_rolling_mean_30min: Optional[float] = None
    vibracao_triaxial_rms_mms_rolling_std_30min: Optional[float] = 0.0
    vibracao_triaxial_rms_mms_rolling_mean_2h: Optional[float] = None
    vibracao_triaxial_rms_mms_rolling_std_2h: Optional[float] = 0.0
    torque_fuso_nm_rolling_mean_30min: Optional[float] = None
    torque_fuso_nm_rolling_std_30min: Optional[float] = 0.0
    torque_fuso_nm_rolling_mean_2h: Optional[float] = None
    torque_fuso_nm_rolling_std_2h: Optional[float] = 0.0

class RespostaPredicao(BaseModel):
    id_celula: str
    tipo_falha: int
    diagnostico: str
    probabilidade_pct: float
    acao_recomendada: str
    impacto_oee: dict

class LoteSensores(BaseModel):
    leituras: List[LeituraSensor]

class RespostaLote(BaseModel):
    total_processado: int
    predicoes: List[RespostaPredicao]

class HealthStatus(BaseModel):
    status: str
    modelo_carregado: bool
    scaler_carregado: bool
    versao: str

def preparar_vetor_features(leitura: LeituraSensor) -> list:
    """Prepara o vetor de 20 features conforme a ordem exata do scaler e do modelo."""
    iem_val = leitura.iem
    if iem_val is None:
        iem_val = (leitura.torque_fuso_nm * leitura.vibracao_triaxial_rms_mms) / (leitura.vazao_refrigerante_lmin + 1.0)
    
    t_mean_30 = leitura.temperatura_fuso_c_rolling_mean_30min or leitura.temperatura_fuso_c
    t_std_30 = leitura.temperatura_fuso_c_rolling_std_30min or 0.0
    t_mean_2h = leitura.temperatura_fuso_c_rolling_mean_2h or leitura.temperatura_fuso_c
    t_std_2h = leitura.temperatura_fuso_c_rolling_std_2h or 0.0
    
    v_mean_30 = leitura.vibracao_triaxial_rms_mms_rolling_mean_30min or leitura.vibracao_triaxial_rms_mms
    v_std_30 = leitura.vibracao_triaxial_rms_mms_rolling_std_30min or 0.0
    v_mean_2h = leitura.vibracao_triaxial_rms_mms_rolling_mean_2h or leitura.vibracao_triaxial_rms_mms
    v_std_2h = leitura.vibracao_triaxial_rms_mms_rolling_std_2h or 0.0
    
    torq_mean_30 = leitura.torque_fuso_nm_rolling_mean_30min or leitura.torque_fuso_nm
    torq_std_30 = leitura.torque_fuso_nm_rolling_std_30min or 0.0
    torq_mean_2h = leitura.torque_fuso_nm_rolling_mean_2h or leitura.torque_fuso_nm
    torq_std_2h = leitura.torque_fuso_nm_rolling_std_2h or 0.0

    return [
        leitura.temperatura_fuso_c,
        leitura.vibracao_triaxial_rms_mms,
        leitura.torque_fuso_nm,
        leitura.vazao_refrigerante_lmin,
        leitura.corrente_estator_amp,
        leitura.pressao_hidraulica_bar,
        leitura.nivel_ruido_db,
        iem_val,
        t_mean_30, t_std_30, t_mean_2h, t_std_2h,
        v_mean_30, v_std_30, v_mean_2h, v_std_2h,
        torq_mean_30, torq_std_30, torq_mean_2h, torq_std_2h
    ]

# Endpoints
@app.get("/health", response_model=HealthStatus, tags=["Monitoramento"])
def health_check():
    """Checa a saúde da API e a disponibilidade dos modelos de Machine Learning."""
    return HealthStatus(
        status="healthy" if (modelo is not None and scaler is not None) else "degraded",
        modelo_carregado=(modelo is not None),
        scaler_carregado=(scaler is not None),
        versao="1.0.0"
    )

@app.post("/predict", response_model=RespostaPredicao, tags=["Inferência Industrial"])
def predict(leitura: LeituraSensor):
    """Diagnóstico em tempo real para um único pacote de sensores de uma célula."""
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos de IA não carregados no servidor.")
    
    vetor = preparar_vetor_features(leitura)
    vetor_df = pd.DataFrame([vetor], columns=FEATURES_ORDEM)
    vetor_scaled = scaler.transform(vetor_df)
    
    classe = int(modelo.predict(vetor_scaled)[0])
    prob = float(np.max(modelo.predict_proba(vetor_scaled)[0]) * 100)
    
    return RespostaPredicao(
        id_celula=leitura.id_celula,
        tipo_falha=classe,
        diagnostico=MAPA_DIAGNOSTICOS.get(classe, "Falha Desconhecida"),
        probabilidade_pct=round(prob, 2),
        acao_recomendada=MAPA_ACOES.get(classe, "AVALIAR_MANUALMENTE"),
        impacto_oee=calcular_oee(classe)
    )

@app.post("/predict-batch", response_model=RespostaLote, tags=["Inferência Industrial"])
def predict_batch(lote: LoteSensores):
    """Inferência em lote para múltiplos registros de sensores simultâneos."""
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos de IA não carregados no servidor.")
    
    vetores = [preparar_vetor_features(l) for l in lote.leituras]
    vetores_df = pd.DataFrame(vetores, columns=FEATURES_ORDEM)
    vetores_scaled = scaler.transform(vetores_df)
    
    classes = modelo.predict(vetores_scaled)
    probas = modelo.predict_proba(vetores_scaled)
    
    resultados = []
    for i, leitura in enumerate(lote.leituras):
        cls_int = int(classes[i])
        prob_val = float(np.max(probas[i]) * 100)
        resultados.append(RespostaPredicao(
            id_celula=leitura.id_celula,
            tipo_falha=cls_int,
            diagnostico=MAPA_DIAGNOSTICOS.get(cls_int, "Falha Desconhecida"),
            probabilidade_pct=round(prob_val, 2),
            acao_recomendada=MAPA_ACOES.get(cls_int, "AVALIAR_MANUALMENTE"),
            impacto_oee=calcular_oee(cls_int)
        ))

    return RespostaLote(
        total_processado=len(resultados),
        predicoes=resultados
    )

@app.post("/vision/inspect-weld", tags=["Visão Computacional"])
async def inspect_weld(file: UploadFile = File(...)):
    """Recebe uma foto do cordão de solda e analisa porosidade e descontinuidade geométrica."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if imagem is None:
        raise HTTPException(status_code=400, detail="Formato de imagem inválido.")
    
    _, relatorio = inspecionar_cordao_solda(imagem)
    return relatorio

@app.post("/vision/inspect-haz", tags=["Visão Computacional"])
async def inspect_haz(file: UploadFile = File(...)):
    """Recebe uma imagem térmica e analisa a Zona Afetada pelo Calor (HAZ) via HSV."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if imagem is None:
        raise HTTPException(status_code=400, detail="Formato de imagem térmica inválido.")
    
    _, _, relatorio_haz = analisar_termografia_haz(imagem)
    return relatorio_haz