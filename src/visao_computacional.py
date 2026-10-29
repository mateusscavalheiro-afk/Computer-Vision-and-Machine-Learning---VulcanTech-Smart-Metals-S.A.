import cv2
import numpy as np
from typing import Dict, Any, Tuple

def aplicar_filtro_nitidez(imagem: np.ndarray) -> np.ndarray:
    """Aplica kernel de nitidez para destacar descontinuidades no cordão."""
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    return cv2.filter2D(imagem, -1, kernel)

def inspecionar_cordao_solda(
    caminho_ou_imagem,
    limiar_canny_baixo: int = 50,
    limiar_canny_alto: int = 150,
    area_minima_porosidade: float = 15.0,
    area_maxima_porosidade: float = 800.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Inspeção óptica e geométrica do cordão de solda robótica (Canny + Contornos)."""
    if isinstance(caminho_ou_imagem, str):
        imagem = cv2.imread(caminho_ou_imagem)
    else:
        imagem = caminho_ou_imagem.copy()

    # 1. Filtro de nitidez e conversão para cinza
    img_nitida = aplicar_filtro_nitidez(imagem)
    img_gray = cv2.cvtColor(img_nitida, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 2. Canny Edge Detection
    bordas = cv2.Canny(img_blur, limiar_canny_baixo, limiar_canny_alto)
    kernel_morf = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bordas_fechadas = cv2.morphologyEx(bordas, cv2.MORPH_CLOSE, kernel_morf)

    # 3. Análise de contornos de porosidade
    contornos, _ = cv2.findContours(bordas_fechadas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    defeitos = []
    area_defeitos = 0.0
    img_resultado = imagem.copy()

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area_minima_porosidade <= area <= area_maxima_porosidade:
            defeitos.append(cnt)
            area_defeitos += area
            cv2.drawContours(img_resultado, [cnt], -1, (0, 0, 255), 2)

    total_px = imagem.shape[0] * imagem.shape[1]
    indice_porosidade = (area_defeitos / total_px) * 100.0
    aprovado = bool(len(defeitos) <= 3 and indice_porosidade < 1.5)

    relatorio = {
        "status_aprovacao": "APROVADO" if aprovado else "REPROVADO",
        "total_porosidades_detectadas": len(defeitos),
        "area_defeitos_px": round(area_defeitos, 2),
        "indice_porosidade_pct": round(indice_porosidade, 4),
        "acao_recomendada": "LIBERAR_LOTE" if aprovado else "REINSPECIONAR_OU_RETRABALHO"
    }
    return img_resultado, relatorio

def analisar_termografia_haz(
    caminho_ou_imagem,
    limiar_s_baixo: int = 140,
    limiar_v_baixo: int = 180
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Análise térmica da Zona Afetada pelo Calor (HAZ) via espaço de cor HSV."""
    if isinstance(caminho_ou_imagem, str):
        imagem = cv2.imread(caminho_ou_imagem)
    else:
        imagem = caminho_ou_imagem.copy()

    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    # Faixas de sobreaquecimento crítico (tons quentes de alta saturação/brilho)
    mascara1 = cv2.inRange(hsv, np.array([0, limiar_s_baixo, limiar_v_baixo]), np.array([15, 255, 255]))
    mascara2 = cv2.inRange(hsv, np.array([170, limiar_s_baixo, limiar_v_baixo]), np.array([180, 255, 255]))
    mascara_hotspot = cv2.bitwise_or(mascara1, mascara2)

    area_critica = float(np.sum(mascara_hotspot > 0))
    proporcao_haz = (area_critica / (imagem.shape[0] * imagem.shape[1])) * 100.0
    alerta = bool(proporcao_haz > 3.0)

    relatorio_haz = {
        "alerta_termico_haz": "CRITICO_SOBREAQUECIDO" if alerta else "NORMAL_ESTAVEL",
        "proporcao_critica_pct": round(proporcao_haz, 2),
        "acao_prescritiva": "REDUZIR_CORRENTE_SOLDA" if alerta else "PARAMETROS_OTIMIZADOS"
    }
    return imagem, mascara_hotspot, relatorio_haz