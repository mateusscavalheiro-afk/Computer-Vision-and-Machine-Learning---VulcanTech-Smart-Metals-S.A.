import os
from typing import List, Optional, Dict, Any
import joblib
import numpy as np
import pandas as pd
import cv2
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
    title="VulcanTech Smart Metals S.A. - Telemetria, OEE e Visão Computacional",
    description="API RESTful em Malha Fechada com Diagnóstico Prescritivo de Engenharia e Inspeção Óptica.",
    version="2.0.0"
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

MAPA_DIAGNOSTICOS = {
    0: "0 - Operação Normal Estável",
    1: "1 - Desgaste Severo de Ferramenta / Fuso",
    2: "2 - Obstrução no Sistema de Refrigeração / Superaquecimento",
    3: "3 - Desalinhamento da Matriz de Prensa / Trinca Mecânica"
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


def calcular_oee(classe_predita: int) -> Dict[str, float]:
    """Calcula o OEE projetado conforme o modo de falha predito."""
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


def gerar_laudo_engenharia(classe: int, leitura: "LeituraSensor", iem_calculado: float) -> Dict[str, Any]:
    """Gera análise técnica completa do problema simulado, causa-raiz e procedimento de reparo."""
    anomalias_detectadas = []
    if leitura.temperatura_fuso_c > 75.0:
        anomalias_detectadas.append(f"Temperatura elevada ({leitura.temperatura_fuso_c}°C > máx 75°C)")
    if leitura.vibracao_triaxial_rms_mms > 3.8:
        anomalias_detectadas.append(f"Vibração RMS crítica ({leitura.vibracao_triaxial_rms_mms} mm/s > máx 3.8 mm/s)")
    if leitura.torque_fuso_nm > 180.0:
        anomalias_detectadas.append(f"Sobrecarga de torque ({leitura.torque_fuso_nm} Nm)")
    if leitura.vazao_refrigerante_lmin < 15.0:
        anomalias_detectadas.append(f"Queda de vazão refrigerante ({leitura.vazao_refrigerante_lmin} L/min < mín 15 L/min)")
    if leitura.corrente_estator_amp > 45.0:
        anomalias_detectadas.append(f"Sobrecorrente estatórica ({leitura.corrente_estator_amp} A)")
    if leitura.pressao_hidraulica_bar < 110.0 or leitura.pressao_hidraulica_bar > 220.0:
        anomalias_detectadas.append(f"Desvio de pressão hidráulica ({leitura.pressao_hidraulica_bar} bar)")
    if leitura.nivel_ruido_db > 85.0:
        anomalias_detectadas.append(f"Emissão acústica severa ({leitura.nivel_ruido_db} dB > máx 85 dB)")

    if classe == 0:
        return {
            "severidade": "NOMINAL",
            "iem_calculado": round(iem_calculado, 2),
            "sintomas_detectados": anomalias_detectadas or ["Todas as variáveis dentro da janela operacional nominal."],
            "o_que_esta_errado": "Nenhuma anomalia eletromecânica ou térmica detectada. O Índice de Esforço Mecânico (IEM) opera em regime laminar estável.",
            "causa_raiz_provavel": "Célula operando com lubrificação, geometria de corte e arrefecimento balanceados para a liga metálica selecionada.",
            "como_consertar": [
                "Manter setpoint de avanço e rotação (100% Override) no CLP.",
                "Seguir cronograma padrão de manutenção preventiva e limpeza de cavacos."
            ],
            "comando_clp_detalhado": "DB10.DBX0.0 = TRUE (Ciclo Normal) | Speed_Override = 100%"
        }
    elif classe == 1:
        return {
            "severidade": "ALERTA ALTO (MANUTENÇÃO PROGRAMADA)",
            "iem_calculado": round(iem_calculado, 2),
            "sintomas_detectados": anomalias_detectadas,
            "o_que_esta_errado": (
                f"Atrito mecânico severo na interface ferramenta-peça (IEM = {iem_calculado:.1f}). "
                "Elevação simultânea de torque no fuso, corrente do motor de acionamento e vibração triaxial."
            ),
            "causa_raiz_provavel": (
                "Desgaste de flanco (VB) acentuado ou lascamento da aresta do inserto de usinagem CNC, "
                "somado a possível fadiga inicial nos rolamentos de contato angular do árvore/fuso."
            ),
            "como_consertar": [
                "1. Atuar via CLP reduzindo imediatamente o avanço de corte (Feedrate Override para 50%) para conter quebra catastrófica.",
                "2. No fim do ciclo da peça atual, comandar indexação no magazine para ferramenta irmã (Sister Tool).",
                "3. Inspecionar batimento radial (Run-out) do cone porta-ferramenta e medir folga axial do fuso com relógio comparador."
            ],
            "comando_clp_detalhado": "DB10.DBW2 = 50 (Feed_Override 50%) | M01 (Parada Condicional Fim de Ciclo)"
        }
    elif classe == 2:
        return {
            "severidade": "ALERTA TÉRMICO (RISCO DE TRINCA TÉRMICA)",
            "iem_calculado": round(iem_calculado, 2),
            "sintomas_detectados": anomalias_detectadas,
            "o_que_esta_errado": (
                f"Colapso na troca térmica da zona de corte/estampagem. Temperatura do fuso atingiu {leitura.temperatura_fuso_c}°C "
                f"com vazão de fluido reduzida para {leitura.vazao_refrigerante_lmin} L/min."
            ),
            "causa_raiz_provavel": (
                "Entupimento dos bicos injetores por lama de usinagem/cavacos finos, saturação do filtro de linha "
                "ou cavitação na bomba centrífuga do tanque de emulsão refrigerante."
            ),
            "como_consertar": [
                "1. Reduzir rotação do fuso e avanço em 50% via CLP para diminuir geração instantânea de calor.",
                "2. Acionar válvula solenoide de retrolavagem (Backflush) ou linha secundária de refrigeração de emergência.",
                "3. Desobstruir galerias de refrigeração interna (Through-Spindle Coolant) e verificar pressão de recalque da bomba."
            ],
            "comando_clp_detalhado": "DB10.DBW2 = 50 | Q4.2 = TRUE (Ativar Bomba Auxiliar Refrigeração)"
        }
    else:
        return {
            "severidade": "EMERGÊNCIA CRÍTICA (PARADA IMEDIATA)",
            "iem_calculado": round(iem_calculado, 2),
            "sintomas_detectados": anomalias_detectadas,
            "o_que_esta_errado": (
                f"Choque mecânico estrutural grave! Vibração extrema ({leitura.vibracao_triaxial_rms_mms} mm/s), "
                f"emissão acústica de {leitura.nivel_ruido_db} dB e perda de pressão hidráulica ({leitura.pressao_hidraulica_bar} bar)."
            ),
            "causa_raiz_provavel": (
                "Desalinhamento das colunas-guia da matriz de prensa hidráulica, início de trinca por fadiga na ferramenta "
                "ou perda de estanqueidade na válvula proporcional de retenção do cilindro principal."
            ),
            "como_consertar": [
                "1. DISPARAR INTERTRAVAMENTO DE EMERGÊNCIA (Safe Torque Off - STO) no CLP para travar o cabeçote imediatamente.",
                "2. Despressurizar linha hidráulica com segurança e bloquear equipamento (Procedimento LOTO).",
                "3. Realizar ensaio não-destrutivo por líquido penetrante/ultrassom na matriz para localizar trincas e realinhar colunas-guia."
            ],
            "comando_clp_detalhado": "M00 / E_STOP = TRUE (Corte Imediato de Acionamento) | Hidraulica_Dump = OPEN"
        }


class LeituraSensor(BaseModel):
    id_celula: str = Field(default="CEL-201")
    liga_metalica: Optional[str] = Field(default="Aco_Carbono_316L")
    temperatura_fuso_c: float
    vibracao_triaxial_rms_mms: float
    torque_fuso_nm: float
    vazao_refrigerante_lmin: float
    corrente_estator_amp: float
    pressao_hidraulica_bar: float
    nivel_ruido_db: float
    
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
    analise_engenharia: dict


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


def preparar_vetor_features(leitura: LeituraSensor) -> tuple:
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

    vetor = [
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
    return vetor, iem_val


@app.get("/health", response_model=HealthStatus, tags=["Monitoramento"])
def health_check():
    return HealthStatus(
        status="healthy" if (modelo is not None and scaler is not None) else "degraded",
        modelo_carregado=(modelo is not None),
        scaler_carregado=(scaler is not None),
        versao="2.0.0"
    )


@app.post("/predict", response_model=RespostaPredicao, tags=["Inferência Industrial"])
def predict(leitura: LeituraSensor):
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos de IA não carregados no servidor.")
    
    vetor, iem_val = preparar_vetor_features(leitura)
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
        impacto_oee=calcular_oee(classe),
        analise_engenharia=gerar_laudo_engenharia(classe, leitura, iem_val)
    )


@app.post("/predict-batch", response_model=RespostaLote, tags=["Inferência Industrial"])
def predict_batch(lote: LoteSensores):
    if modelo is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelos de IA não carregados no servidor.")
    
    vetores_iem = [preparar_vetor_features(l) for l in lote.leituras]
    vetores = [v[0] for v in vetores_iem]
    iems = [v[1] for v in vetores_iem]

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
            impacto_oee=calcular_oee(cls_int),
            analise_engenharia=gerar_laudo_engenharia(cls_int, leitura, iems[i])
        ))

    return RespostaLote(
        total_processado=len(resultados),
        predicoes=resultados
    )


@app.post("/vision/inspect-weld", tags=["Visão Computacional"])
async def inspect_weld(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if imagem is None:
        raise HTTPException(status_code=400, detail="Formato de imagem inválido.")
    
    _, relatorio = inspecionar_cordao_solda(imagem)
    return relatorio


@app.post("/vision/inspect-haz", tags=["Visão Computacional"])
async def inspect_haz(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if imagem is None:
        raise HTTPException(status_code=400, detail="Formato de imagem térmica inválido.")
    
    _, _, relatorio_haz = analisar_termografia_haz(imagem)
    return relatorio_haz