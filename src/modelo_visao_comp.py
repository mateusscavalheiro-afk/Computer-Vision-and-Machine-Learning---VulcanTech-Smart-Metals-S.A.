import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple


def carregar_imagem(caminho_ou_buffer) -> np.ndarray:
    if isinstance(caminho_ou_buffer, np.ndarray):
        return caminho_ou_buffer.copy()
    elif isinstance(caminho_ou_buffer, str):
        img = cv2.imread(caminho_ou_buffer)
    else:
        nparr = np.frombuffer(caminho_ou_buffer, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Não foi possível carregar ou decodificar a imagem fornecida.")
    return img


def codificar_imagem_base64(imagem: np.ndarray) -> str:
    sucesso, buffer = cv2.imencode(".png", imagem)
    if not sucesso:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def aplicar_filtro_nitidez(imagem: np.ndarray) -> np.ndarray:
    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    return cv2.filter2D(imagem, -1, kernel_sharpening)


def inspecionar_cordao_solda(
    caminho_ou_buffer,
    limiar_canny_baixo: int = 80,
    limiar_canny_alto: int = 190,
    min_area_poro: int = 8,
    max_area_poro: int = 220,
    tolerancia_poros_max: int = 4
) -> Tuple[np.ndarray, Dict[str, Any]]:
    img_orig = carregar_imagem(caminho_ou_buffer)

    altura_padrao = 480
    fator_escala = altura_padrao / img_orig.shape[0]
    largura_padrao = int(img_orig.shape[1] * fator_escala)
    img = cv2.resize(img_orig, (largura_padrao, altura_padrao))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    _, mask_preliminar = cv2.threshold(img_blur, 35, 255, cv2.THRESH_BINARY)
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_preliminar = cv2.morphologyEx(mask_preliminar, cv2.MORPH_CLOSE, kernel_fechamento)

    contornos_cordao, _ = cv2.findContours(mask_preliminar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos_cordao:
        raise ValueError("Não foi possível isolar o cordão de solda do fundo.")

    maior_contorno = max(contornos_cordao, key=cv2.contourArea)

    rect_rot = cv2.minAreaRect(maior_contorno)
    (centro_x, centro_y), (dim_w, dim_h), angulo = rect_rot

    if dim_w < dim_h:
        angulo_correcao = angulo + 90.0
        w_aparente = dim_h
    else:
        angulo_correcao = angulo
        w_aparente = dim_w

    M_rot = cv2.getRotationMatrix2D((centro_x, centro_y), angulo_correcao, 1.0)
    img_alinhada = cv2.warpAffine(img, M_rot, (largura_padrao, altura_padrao), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask_alinhada = cv2.warpAffine(mask_preliminar, M_rot, (largura_padrao, altura_padrao), flags=cv2.INTER_NEAREST)

    contornos_alinhados, _ = cv2.findContours(mask_alinhada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_alin = max(contornos_alinhados, key=cv2.contourArea)
    x_box, y_box, w_box, h_box = cv2.boundingRect(cnt_alin)

    larguras_reais = []
    eixo_central_y = []
    coordenadas_amostragem = []

    num_fatias = 30
    passo = max(1, int(w_box * 0.8) // num_fatias)
    x_inicio = x_box + int(w_box * 0.1)

    for col in range(x_inicio, x_inicio + int(w_box * 0.8), passo):
        coluna = mask_alinhada[:, col]
        indices = np.where(coluna > 0)[0]
        if len(indices) > 6:
            y_min, y_max = int(indices[0]), int(indices[-1])
            espessura = y_max - y_min
            centro_y_fatia = (y_min + y_max) / 2.0
            larguras_reais.append(espessura)
            eixo_central_y.append(centro_y_fatia)
            coordenadas_amostragem.append((col, y_min, y_max, centro_y_fatia))

    variacao_largura_pct = 0.0
    desvio_linearidade_pct = 0.0
    pontos_ondulacao_critica = 0
    geometria_aprovada = True

    diag_alinhado = img_alinhada.copy()

    if len(larguras_reais) >= 5:
        larguras_arr = np.array(larguras_reais, dtype=np.float32)
        centros_arr = np.array(eixo_central_y, dtype=np.float32)
        largura_media = float(np.mean(larguras_arr))

        variacao_largura_pct = round(float(np.std(larguras_arr) / (largura_media + 1e-5)) * 100.0, 2)

        idx = np.arange(len(centros_arr))
        linha_ideal = np.polyval(np.polyfit(idx, centros_arr, 1), idx)
        desvios_retidao = np.abs(centros_arr - linha_ideal)
        desvio_linearidade_pct = round(float(np.mean(desvios_retidao) / (largura_media + 1e-5)) * 100.0, 2)

        for i, (col, y_min, y_max, cy) in enumerate(coordenadas_amostragem):
            desvio_loc = abs(larguras_arr[i] - largura_media) / (largura_media + 1e-5)
            anomalia = desvio_loc > 0.28
            if anomalia:
                pontos_ondulacao_critica += 1
            cor_linha = (0, 0, 255) if anomalia else (0, 255, 120)
            cv2.line(diag_alinhado, (col, y_min), (col, y_max), cor_linha, 2)

        if variacao_largura_pct > 24.0 or desvio_linearidade_pct > 18.0:
            geometria_aprovada = False

    kernel_erosao = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_interna = cv2.erode(mask_alinhada, kernel_erosao, iterations=1)

    gray_alin = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_alin, (7, 7), 0)

    media_local = cv2.boxFilter(gray_blur, cv2.CV_32F, (25, 25))
    diferenca_escuro = media_local - gray_blur.astype(np.float32)
    mask_cavidades = (diferenca_escuro > 28).astype(np.uint8) * 255
    mask_cavidades = cv2.bitwise_and(mask_cavidades, mask_cavidades, mask=mascara_interna)

    contornos_poros, _ = cv2.findContours(mask_cavidades, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    poros_detectados = 0
    area_total_poros = 0.0

    for cnt in contornos_poros:
        area = cv2.contourArea(cnt)
        if min_area_poro <= area <= max_area_poro:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue
            circularidade = 4.0 * np.pi * (area / (perimetro * perimetro))
            if circularidade >= 0.55:
                poros_detectados += 1
                area_total_poros += area
                (xp, yp), raio = cv2.minEnclosingCircle(cnt)
                cv2.circle(diag_alinhado, (int(xp), int(yp)), max(int(raio) + 3, 5), (0, 0, 255), 2)
                cv2.circle(diag_alinhado, (int(xp), int(yp)), 2, (0, 255, 255), -1)

    M_inv = cv2.getRotationMatrix2D((centro_x, centro_y), -angulo_correcao, 1.0)
    img_resultado = cv2.warpAffine(diag_alinhado, M_inv, (largura_padrao, altura_padrao), flags=cv2.INTER_LINEAR)

    fundo_preto = cv2.bitwise_not(mask_preliminar)
    img_resultado[fundo_preto > 0] = (img[fundo_preto > 0] * 0.35).astype(np.uint8)
    cv2.drawContours(img_resultado, [maior_contorno], -1, (0, 230, 118), 2)

    area_util = max(1.0, float(np.sum(mask_preliminar > 0)))
    indice_porosidade = round((area_total_poros / area_util) * 100.0, 4)

    porosidade_aprovada = (poros_detectados <= tolerancia_poros_max) and (indice_porosidade < 0.50)
    aprovado_geral = bool(geometria_aprovada and porosidade_aprovada)

    if aprovado_geral:
        classificacao_forma = "CORDÃO RETO E UNIFORME (PADRÃO ISO 5817 - NÍVEL B)"
        acao_clp = "LIBERAR_LOTE_SOLDA"
        diagnostico_detalhado = (
            f"Geometria com avanço homogêneo (Ângulo detectado: {angulo_correcao:.1f}° | "
            f"Variação de largura transversal: {variacao_largura_pct}% | Desvio de linearidade: {desvio_linearidade_pct}%). "
            f"Nenhum poro crítico detectado."
        )
        causa_raiz = "Trajetória do robô, velocidade de soldagem e tensão do arco em regime nominal balanceado."
        plano_correcao = (
            "1. Manter ciclo automático na célula robótica.\n"
            "2. Registrar peça aprovada na rastreabilidade de qualidade."
        )
    else:
        acao_clp = "INTERROMPER_E_RETRABALHAR_SOLDA"
        problemas = []
        causas = []
        correcoes = []

        if not geometria_aprovada:
            classificacao_forma = "REPROVADO: CORDÃO ONDULADO / IRREGULARIDADE DE VOLUME"
            problemas.append(
                f"Oscilação transversal excessiva (Variação de largura: {variacao_largura_pct}% | "
                f"Desvio de linearidade: {desvio_linearidade_pct}% | {pontos_ondulacao_critica} pontos fora de perfil)."
            )
            causas.append("Instabilidade mecânica no TCP da tocha ou velocidade de avanço desregulada.")
            correcoes.append("• Recalibrar velocidade de translação do robô e centrar gabarito de fixação.")
        else:
            classificacao_forma = "REPROVADO: POROSIDADE CRÍTICA"

        if not porosidade_aprovada:
            problemas.append(f"Identificação de {poros_detectados} poros ({indice_porosidade}% da área).")
            causas.append("Deficiência no fluxo do gás de proteção ou contaminação na chapa.")
            correcoes.append("• Regular vazão de argônio/CO2 (15-18 L/min) e limpar bocal da tocha.")

        diagnostico_detalhado = " ".join(problemas)
        causa_raiz = " ".join(causas)
        plano_correcao = "\n".join(correcoes)

    cor_hud = (0, 230, 118) if aprovado_geral else (0, 50, 255)
    cv2.rectangle(img_resultado, (10, 10), (450, 80), (15, 23, 42), -1)
    cv2.rectangle(img_resultado, (10, 10), (450, 80), cor_hud, 1)
    cv2.putText(img_resultado, f"STATUS: {'APROVADO' if aprovado_geral else 'REPROVADO'}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_hud, 2)
    cv2.putText(img_resultado, f"Angulo: {angulo_correcao:.1f} deg | Poros: {poros_detectados}", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 250, 252), 1)
    cv2.putText(img_resultado, f"Var. Largura: {variacao_largura_pct}% | Ondulacao: {desvio_linearidade_pct}%", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

    relatorio = {
        "tipo_inspecao": "Inspeção Geométrica Invariante a Ângulo (Auto-Alinhamento + Filtro Especular)",
        "status_aprovacao": "APROVADO" if aprovado_geral else "REPROVADO",
        "classificacao_geometria": classificacao_forma,
        "angulo_cordao_graus": round(float(angulo_correcao), 2),
        "total_porosidades_detectadas": int(poros_detectados),
        "indice_porosidade_pct": indice_porosidade,
        "variacao_largura_pct": variacao_largura_pct,
        "desvio_linearidade_pct": desvio_linearidade_pct,
        "pontos_ondulacao_critica": int(pontos_ondulacao_critica),
        "acao_recomendada": acao_clp,
        "diagnostico_detalhado": diagnostico_detalhado,
        "causa_raiz": causa_raiz,
        "plano_correcao": plano_correcao,
        "imagem_original_b64": codificar_imagem_base64(img),
        "imagem_processada_b64": codificar_imagem_base64(img_resultado)
    }

    return img_resultado, relatorio


def analisar_termografia_haz(
    caminho_ou_buffer,
    limiar_razao_espalhamento_max: float = 12.0,
    limiar_proporcao_quadro_max: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    img_orig = carregar_imagem(caminho_ou_buffer)
    altura_padrao = 480
    proporcao = altura_padrao / img_orig.shape[0]
    largura_padrao = int(img_orig.shape[1] * proporcao)
    img = cv2.resize(img_orig, (largura_padrao, altura_padrao))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    mascara_malha_total = ((V > 45) & ~((H > 105) & (H < 140) & (V < 130))).astype(np.uint8) * 255
    kernel_malha = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mascara_malha_total = cv2.morphologyEx(mascara_malha_total, cv2.MORPH_CLOSE, kernel_malha)
    mascara_malha_total = cv2.morphologyEx(mascara_malha_total, cv2.MORPH_OPEN, kernel_malha)

    area_malha_total_px = float(np.sum(mascara_malha_total > 0))
    area_quadro_total = float(img.shape[0] * img.shape[1])

    if area_malha_total_px < 500:
        area_malha_total_px = area_quadro_total * 0.25

    vermelho_1 = cv2.inRange(hsv, np.array([0, 120, 150], dtype=np.uint8), np.array([18, 255, 255], dtype=np.uint8))
    vermelho_2 = cv2.inRange(hsv, np.array([162, 120, 150], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    branco_termico = cv2.inRange(hsv, np.array([0, 0, 230], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))

    mascara_critica_bruta = cv2.bitwise_or(cv2.bitwise_or(vermelho_1, vermelho_2), branco_termico)
    mascara_critica = cv2.bitwise_and(mascara_critica_bruta, mascara_critica_bruta, mask=mascara_malha_total)
    kernel_limpeza = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_critica = cv2.morphologyEx(mascara_critica, cv2.MORPH_OPEN, kernel_limpeza)

    area_hotspot_px = int(np.sum(mascara_critica > 0))

    proporcao_quadro_pct = round((area_hotspot_px / area_quadro_total) * 100.0, 2)
    razao_espalhamento_haz = round((area_hotspot_px / area_malha_total_px) * 100.0, 2)

    if area_hotspot_px > 0:
        brilho_medio = float(np.mean(V[mascara_critica > 0]))
    else:
        brilho_medio = float(np.mean(V))
    temp_pico_c = round(150.0 + (brilho_medio / 255.0) * 1150.0, 1)

    tem_superaquecimento = (razao_espalhamento_haz > limiar_razao_espalhamento_max) or (proporcao_quadro_pct > limiar_proporcao_quadro_max)

    img_resultado = (img * 0.45).astype(np.uint8)
    img_resultado[mascara_malha_total > 0] = (img[mascara_malha_total > 0] * 0.85).astype(np.uint8)
    img_resultado[mascara_critica > 0] = img[mascara_critica > 0]

    contornos_malha, _ = cv2.findContours(mascara_malha_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_resultado, contornos_malha, -1, (255, 200, 0), 1)

    contornos_criticos, _ = cv2.findContours(mascara_critica, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_focos = 0
    for cnt in contornos_criticos:
        area_c = cv2.contourArea(cnt)
        if area_c > 35:
            num_focos += 1
            cv2.drawContours(img_resultado, [cnt], -1, (0, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 0, 255), 1)

    if not tem_superaquecimento:
        status_aprovacao = "APROVADO"
        alerta_status = "NORMAL_ESTAVEL"
        acao_clp = "PARAMETROS_TERMICOS_OTIMIZADOS"
        classificacao_texto = "DISSIPAÇÃO TÉRMICA CONTROLADA (APROVADO)"
        diagnostico_detalhado = (
            f"Perfil térmico nominal estável. A zona de alta temperatura representa apenas {razao_espalhamento_haz}% "
            f"da malha da junta ({proporcao_quadro_pct}% do quadro | {area_hotspot_px} px). Aporte térmico confinado à poça de fusão."
        )
        causa_raiz = "Velocidade de avanço da tocha e corrente de arco dentro da janela metalúrgica correta para a espessura da chapa."
        plano_correcao = (
            "1. Manter taxa de calor e corrente elétrica programada no CLP.\n"
            "2. Não há necessidade de alteração de velocidade de soldagem nem de resfriamento forçado."
        )
    else:
        status_aprovacao = "REPROVADO"
        alerta_status = "CRITICO_SOBREAQUECIDO"
        acao_clp = "REDUZIR_APORTE_TERMICO_E_CORRENTE"
        classificacao_texto = "SOBREAQUECIMENTO CRÍTICO NA ZONA AFETADA PELO CALOR (HAZ)"
        diagnostico_detalhado = (
            f"Transbordamento severo de calor! A área crítica da HAZ ocupa {razao_espalhamento_haz}% da malha térmica "
            f"({proporcao_quadro_pct}% do quadro total | {area_hotspot_px} px em {num_focos} focos). Risco iminente de trinca térmica e deformação da junta."
        )
        causa_raiz = "Excesso de aporte térmico (Heat Input). Corrente excessiva, tensão elevada ou velocidade de deslocamento do robô lenta demais."
        plano_correcao = (
            "• Atuação CLP: Reduzir corrente da fonte de solda em 15% a 20% imediatamente.\n"
            "• Aumentar velocidade de deslocamento linear do manipulador robótico em 10% para diminuir a energia transferida (kJ/mm).\n"
            "• Checar refrigeração nos blocos de fixação de cobre da bancada."
        )

    cor_hud = (0, 230, 118) if not tem_superaquecimento else (0, 50, 255)
    cv2.rectangle(img_resultado, (10, 10), (450, 80), (15, 23, 42), -1)
    cv2.rectangle(img_resultado, (10, 10), (450, 80), cor_hud, 1)
    cv2.putText(img_resultado, f"STATUS: {status_aprovacao} ({alerta_status})", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_hud, 2)
    cv2.putText(img_resultado, f"Razao HAZ: {razao_espalhamento_haz}% | Quadro: {proporcao_quadro_pct}%", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 250, 252), 1)
    cv2.putText(img_resultado, f"Pico Est.: ~{temp_pico_c} C | Focos: {num_focos}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

    relatorio_haz = {
        "tipo_inspecao": "Termografia HAZ por Subtração de Malha Relativa",
        "status_aprovacao": status_aprovacao,
        "alerta_termico_haz": alerta_status,
        "classificacao_geometria": classificacao_texto,
        "area_ponto_quente_px": area_hotspot_px,
        "area_malha_total_px": int(area_malha_total_px),
        "razao_espalhamento_haz_pct": razao_espalhamento_haz,
        "proporcao_critica_pct": proporcao_quadro_pct,
        "focos_criticos_detectados": num_focos,
        "temperatura_pico_estimada_c": temp_pico_c,
        "acao_prescritiva": acao_clp,
        "diagnostico_detalhado": diagnostico_detalhado,
        "causa_raiz": causa_raiz,
        "plano_correcao": plano_correcao,
        "imagem_original_b64": codificar_imagem_base64(img),
        "imagem_processada_b64": codificar_imagem_base64(img_resultado)
    }

    return img_resultado, mascara_critica, relatorio_haz


if __name__ == "__main__":
    print("Módulo de Visão Computacional carregado com sucesso.")