import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple


def carregar_imagem(caminho_ou_buffer) -> np.ndarray:
    """Carrega imagem de caminho em string, bytes ou numpy array."""
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
    """Converte uma matriz de imagem OpenCV (BGR) para string Base64 (Data URI)."""
    sucesso, buffer = cv2.imencode(".png", imagem)
    if not sucesso:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def aplicar_filtro_nitidez(imagem: np.ndarray) -> np.ndarray:
    """Aplica kernel de convolução passa-alta para realçar bordas, trincas e porosidades."""
    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    return cv2.filter2D(imagem, -1, kernel_sharpening)

def inspecionar_cordao_solda(
    caminho_ou_buffer,
    limiar_canny_baixo: int = 45,
    limiar_canny_alto: int = 140,
    min_area_poro: int = 4,
    max_area_poro: int = 280,
    tolerancia_poros_max: int = 3
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Realiza a inspeção óptica avançada de cordões de solda (preferencialmente sobre fundo escuro/preto):
    - Segmenta o cordão de solda do fundo preto e calcula métricas de geometria transversal.
    - Avalia se o cordão está reto/uniforme (BOM) ou excessivamente ondulado/volumoso (RUIM).
    - Aplica Gradiente Morfológico, Black-Hat e Canny Edge Detection para isolar porosidades reais.
    - Gera visualização analítica com mapa de gradiente, desvios geométricos e marcação dos pontos críticos.
    """
    img_orig = carregar_imagem(caminho_ou_buffer)
    # Redimensionamento padronizado para manter consistência de métricas geométricas
    altura_padrao = 480
    proporcao = altura_padrao / img_orig.shape[0]
    largura_padrao = int(img_orig.shape[1] * proporcao)
    img = cv2.resize(img_orig, (largura_padrao, altura_padrao))
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_gray)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    # 1. Segmentação do Cordão sobre Fundo Preto
    # Utiliza Otsu combinado com limiar mínimo para separar o metal refletivo do fundo preto
    limiar_otsu, mascara_cordao = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if limiar_otsu < 25:
        _, mascara_cordao = cv2.threshold(img_blur, 30, 255, cv2.THRESH_BINARY)

    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara_cordao = cv2.morphologyEx(mascara_cordao, cv2.MORPH_CLOSE, kernel_fechamento)
    mascara_cordao = cv2.morphologyEx(mascara_cordao, cv2.MORPH_OPEN, kernel_fechamento)

    # 2. Extração de Gradiente Sobel (Textura e Relevo do Cordão)
    grad_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude_grad = cv2.magnitude(grad_x, grad_y)
    magnitude_grad = cv2.normalize(magnitude_grad, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Mapa de calor do gradiente restrito à região do cordão
    mapa_gradiente = cv2.applyColorMap(magnitude_grad, cv2.COLORMAP_JET)
    img_resultado = cv2.addWeighted(img, 0.65, mapa_gradiente, 0.35, 0)
    # Mantém o fundo preto limpo na imagem de diagnóstico
    fundo_inv = cv2.bitwise_not(mascara_cordao)
    img_resultado[fundo_inv > 0] = (img[fundo_inv > 0] * 0.4).astype(np.uint8)

    # 3. Análise Geométrica do Cordão (Linearidade, Ondulação e Volume)
    contornos_cordao, _ = cv2.findContours(mascara_cordao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    desvio_linearidade_pct = 0.0
    variacao_largura_pct = 0.0
    indice_volume_convexidade = 100.0
    largura_media_px = 0.0
    pontos_ondulacao_critica = 0
    geometria_aprovada = True

    if contornos_cordao:
        maior_contorno = max(contornos_cordao, key=cv2.contourArea)
        area_cordao = cv2.contourArea(maior_contorno)
        hull = cv2.convexHull(maior_contorno)
        area_hull = cv2.contourArea(hull)
        if area_hull > 0:
            indice_volume_convexidade = round((area_cordao / area_hull) * 100.0, 2)

        # Determina orientação principal do cordão (Horizontal vs Vertical)
        x_box, y_box, w_box, h_box = cv2.boundingRect(maior_contorno)
        horizontal = w_box >= h_box

        larguras = []
        centros = []
        coordenadas_eixo = []

        # Fatiamento transversal (Slices) para medir espessura e eixo central
        num_fatias = 40
        if horizontal:
            passo = max(1, w_box // num_fatias)
            for col in range(x_box + passo, x_box + w_box - passo, passo):
                coluna_mask = mascara_cordao[:, col]
                indices_y = np.where(coluna_mask > 0)[0]
                if len(indices_y) > 5:
                    y_min, y_max = int(indices_y[0]), int(indices_y[-1])
                    espessura = y_max - y_min
                    centro_y = (y_min + y_max) / 2.0
                    larguras.append(espessura)
                    centros.append(centro_y)
                    coordenadas_eixo.append((col, int(centro_y), y_min, y_max))
        else:
            passo = max(1, h_box // num_fatias)
            for lin in range(y_box + passo, y_box + h_box - passo, passo):
                linha_mask = mascara_cordao[lin, :]
                indices_x = np.where(linha_mask > 0)[0]
                if len(indices_x) > 5:
                    x_min, x_max = int(indices_x[0]), int(indices_x[-1])
                    espessura = x_max - x_min
                    centro_x = (x_min + x_max) / 2.0
                    larguras.append(espessura)
                    centros.append(centro_x)
                    coordenadas_eixo.append((int(centro_x), lin, x_min, x_max))

        if len(larguras) >= 5:
            larguras_arr = np.array(larguras, dtype=np.float32)
            centros_arr = np.array(centros, dtype=np.float32)
            largura_media_px = float(np.mean(larguras_arr))
            
            # Coeficiente de variação da largura (mede se está muito volumoso/irregular)
            variacao_largura_pct = round(float(np.std(larguras_arr) / (largura_media_px + 1e-5)) * 100.0, 2)

            # Regressão linear no eixo central para medir ondulação (desvio de retidão)
            eixo_idx = np.arange(len(centros_arr))
            coef = np.polyfit(eixo_idx, centros_arr, 1)
            eixo_ideal = np.polyval(coef, eixo_idx)
            desvios_eixo = np.abs(centros_arr - eixo_ideal)
            desvio_linearidade_pct = round(float(np.mean(desvios_eixo) / (largura_media_px + 1e-5)) * 100.0, 2)

            # Desenha as linhas de inspeção geométrica e destaca anomalias de volume/ondulação
            for i, info in enumerate(coordenadas_eixo):
                desvio_larg = abs(larguras_arr[i] - largura_media_px) / (largura_media_px + 1e-5)
                desvio_pos = desvios_eixo[i] / (largura_media_px + 1e-5)

                anomalia_fatia = (desvio_larg > 0.22) or (desvio_pos > 0.18)
                cor_fatia = (0, 140, 255) if anomalia_fatia else (0, 255, 120)

                if anomalia_fatia:
                    pontos_ondulacao_critica += 1

                if horizontal:
                    col, c_y, y_min, y_max = info
                    cv2.line(img_resultado, (col, y_min), (col, y_max), cor_fatia, 1)
                    cv2.circle(img_resultado, (col, int(eixo_ideal[i])), 2, (255, 255, 0), -1)
                    if anomalia_fatia:
                        cv2.circle(img_resultado, (col, y_min), 4, (0, 0, 255), -1)
                        cv2.circle(img_resultado, (col, y_max), 4, (0, 0, 255), -1)
                else:
                    c_x, lin, x_min, x_max = info
                    cv2.line(img_resultado, (x_min, lin), (x_max, lin), cor_fatia, 1)
                    cv2.circle(img_resultado, (int(eixo_ideal[i]), lin), 2, (255, 255, 0), -1)
                    if anomalia_fatia:
                        cv2.circle(img_resultado, (x_min, lin), 4, (0, 0, 255), -1)
                        cv2.circle(img_resultado, (x_max, lin), 4, (0, 0, 255), -1)

        # Desenha o contorno externo do cordão (Verde = Nominal, Amarelo = Contorno)
        cv2.drawContours(img_resultado, [maior_contorno], -1, (0, 230, 118), 2)

        # Critérios de aprovação geométrica (Retidão e Volume)
        if variacao_largura_pct > 18.0 or desvio_linearidade_pct > 14.0 or indice_volume_convexidade < 86.0:
            geometria_aprovada = False

    # 4. Detecção de Porosidades Internas e Mordeduras (Black-Hat + Canny)
    # Erosão na máscara do cordão para buscar poros estritamente dentro do cordão
    kernel_erosao = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mascara_interna = cv2.erode(mascara_cordao, kernel_erosao, iterations=1)

    img_nitida = aplicar_filtro_nitidez(img_eq)
    
    # Operação Black-Hat destaca cavidades escuras (poros e crateras) sobre o metal claro do cordão
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(img_nitida, cv2.MORPH_BLACKHAT, kernel_bh)
    _, thresh_bh = cv2.threshold(blackhat, 32, 255, cv2.THRESH_BINARY)

    # Canny Edge Detection para capturar descontinuidades abruptas
    bordas_canny = cv2.Canny(img_blur, limiar_canny_baixo, limiar_canny_alto)
    comb_defeitos = cv2.bitwise_or(thresh_bh, bordas_canny)
    comb_defeitos = cv2.bitwise_and(comb_defeitos, comb_defeitos, mask=mascara_interna)

    kernel_morf_poro = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    comb_defeitos = cv2.morphologyEx(comb_defeitos, cv2.MORPH_CLOSE, kernel_morf_poro)

    contornos_internos, _ = cv2.findContours(comb_defeitos, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    poros_detectados = 0
    area_total_poros = 0.0

    for cnt in contornos_internos:
        area = cv2.contourArea(cnt)
        if min_area_poro <= area <= max_area_poro:
            perimetro = cv2.arcLength(cnt, True)
            if perimetro == 0:
                continue
            circularidade = 4 * np.pi * (area / (perimetro * perimetro))
            # Filtra cavidades reais (poros e microtrincas de cratera) ignorando linhas normais de escama
            if circularidade >= 0.25:
                poros_detectados += 1
                area_total_poros += area
                (x_p, y_p), raio = cv2.minEnclosingCircle(cnt)
                centro_pt = (int(x_p), int(y_p))
                r_int = max(int(raio) + 4, 6)
                # Marcação de falha estilo AeroMotion-Dynamics (Alvo Vermelho Neon)
                cv2.circle(img_resultado, centro_pt, r_int, (0, 0, 255), 2)
                cv2.circle(img_resultado, centro_pt, 2, (0, 255, 255), -1)

    area_util_cordao = max(1.0, float(np.sum(mascara_cordao > 0)))
    indice_porosidade = round((area_total_poros / area_util_cordao) * 100.0, 4)

    porosidade_aprovada = (poros_detectados <= tolerancia_poros_max) and (indice_porosidade < 0.85)
    aprovado_geral = bool(geometria_aprovada and porosidade_aprovada)

    # Classificação da Morfologia e Diagnóstico de Engenharia
    if aprovado_geral:
        classificacao_forma = "CORDÃO RETO E UNIFORME (PADRÃO ISO 5817 - NÍVEL B)"
        acao_clp = "LIBERAR_LOTE_SOLDA"
        diagnostico_detalhado = (
            "Geometria linear estável com penetração consistente e largura transversal uniforme. "
            "Ausência de porosidade crítica ou mordeduras nas margens."
        )
        causa_raiz = "Parâmetros de arco MIG/MAG, vazão de gás de proteção e velocidade de deslocamento do robô dentro da janela nominal."
        plano_correcao = (
            "1. Manter ciclo automático na célula robótica.\n"
            "2. Registrar lote como aprovado no controle estatístico de processo (CEP)."
        )
    else:
        acao_clp = "INTERROMPER_E_RETRABALHAR_SOLDA"
        problemas = []
        causas = []
        correcoes = []

        if not geometria_aprovada:
            classificacao_forma = "REPROVADO: CORDÃO ONDULADO / EXCESSO DE VOLUME"
            problemas.append(
                f"Instabilidade geométrica severa (Variação de largura: {variacao_largura_pct}% | "
                f"Desvio de linearidade: {desvio_linearidade_pct}% | {pontos_ondulacao_critica} secções críticas fora de perfil)."
            )
            causas.append(
                "Velocidade de soldagem oscilante, desalinhamento da junta na fixação ou tensão de arco incompatível com a taxa de alimentação de arame (gerando cordão abaulado/volumoso)."
            )
            correcoes.append(
                "• Recalibrar trajetória TCP (Tool Center Point) do braço robótico e reduzir oscilação (weaving).\n"
                "• Ajustar relação Tensão (V) vs. Velocidade de Arame (m/min) no programa do CLP para achatar o reforço excessivo."
            )
        else:
            classificacao_forma = "REPROVADO: POROSIDADE SUPERFICIAL / GASES RETIDOS"

        if not porosidade_aprovada:
            problemas.append(
                f"Detecção de {poros_detectados} cavidades/porosidades ({indice_porosidade}% da área do cordão comprometida)."
            )
            causas.append(
                "Falha na cobertura do gás de proteção (Ar/CO2), bocal da tocha obstruído por respingos ou contaminação superficial (óleo/óxido) na chapa base."
            )
            correcoes.append(
                "• Inspecionar válvula proporcional de gás e elevar vazão para 15–18 L/min.\n"
                "• Executar ciclo de limpeza mecânica do bocal da tocha (Reamer) e verificar purga da linha."
            )

        diagnostico_detalhado = " ".join(problemas)
        causa_raiz = " ".join(causas)
        plano_correcao = "\n".join(correcoes)

    # HUD Informativo sobre a imagem processada
    cor_hud = (0, 230, 118) if aprovado_geral else (0, 50, 255)
    cv2.rectangle(img_resultado, (10, 10), (430, 78), (15, 23, 42), -1)
    cv2.rectangle(img_resultado, (10, 10), (430, 78), cor_hud, 1)
    cv2.putText(img_resultado, f"STATUS: {'APROVADO' if aprovado_geral else 'REPROVADO'}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_hud, 2)
    cv2.putText(img_resultado, f"Poros: {poros_detectados} | Var. Largura: {variacao_largura_pct}%", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 250, 252), 1)
    cv2.putText(img_resultado, f"Ondulacao: {desvio_linearidade_pct}% | Convex: {indice_volume_convexidade}%", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

    relatorio = {
        "tipo_inspecao": "Inspeção Geométrica & Porosidade (Gradiente + Canny)",
        "status_aprovacao": "APROVADO" if aprovado_geral else "REPROVADO",
        "classificacao_geometria": classificacao_forma,
        "total_porosidades_detectadas": int(poros_detectados),
        "indice_porosidade_pct": indice_porosidade,
        "variacao_largura_pct": variacao_largura_pct,
        "desvio_linearidade_pct": desvio_linearidade_pct,
        "indice_convexidade_pct": indice_volume_convexidade,
        "pontos_ondulacao_critica": int(pontos_ondulacao_critica),
        "acao_recomendada": acao_clp,
        "diagnostico_detalhado": diagnostico_detalhado,
        "causa_raiz": causa_raiz,
        "plano_correcao": plano_correcao,
        "imagem_original_b64": codificar_imagem_base64(img),
        "imagem_processada_b64": codificar_imagem_base64(img_resultado)
    }

    return img_resultado, relatorio


# =====================================================================
# 2. ANÁLISE TERMOGRÁFICA DA ZONA AFETADA PELO CALOR (HAZ - HSV)
# =====================================================================
def analisar_termografia_haz(
    caminho_ou_buffer,
    limiar_area_critica: int = 350,
    limiar_s_baixo: int = 110,
    limiar_v_baixo: int = 145
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Processa imagens termográficas convertendo para o espaço de cores HSV,
    isolando isotermas críticas na Zona Afetada pelo Calor (HAZ) e gerando mapa visual comparativo.
    """
    img_orig = carregar_imagem(caminho_ou_buffer)
    altura_padrao = 480
    proporcao = altura_padrao / img_orig.shape[0]
    largura_padrao = int(img_orig.shape[1] * proporcao)
    img = cv2.resize(img_orig, (largura_padrao, altura_padrao))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Faixas HSV correspondentes a zonas de sobreaquecimento severo (Vermelho/Laranja/Amarelo Brilhante/Branco Térmico)
    vermelho_baixo_1 = np.array([0, limiar_s_baixo, limiar_v_baixo], dtype=np.uint8)
    vermelho_alto_1 = np.array([22, 255, 255], dtype=np.uint8)
    vermelho_baixo_2 = np.array([160, limiar_s_baixo, limiar_v_baixo], dtype=np.uint8)
    vermelho_alto_2 = np.array([180, 255, 255], dtype=np.uint8)
    
    mascara_1 = cv2.inRange(hsv, vermelho_baixo_1, vermelho_alto_1)
    mascara_2 = cv2.inRange(hsv, vermelho_baixo_2, vermelho_alto_2)
    mascara_hotspots = cv2.bitwise_or(mascara_1, mascara_2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_limpa = cv2.morphologyEx(mascara_hotspots, cv2.MORPH_OPEN, kernel)
    mascara_limpa = cv2.morphologyEx(mascara_limpa, cv2.MORPH_CLOSE, kernel)
    
    area_hotspot_px = int(np.sum(mascara_limpa > 0))
    area_total = img.shape[0] * img.shape[1]
    proporcao_haz = round((area_hotspot_px / area_total) * 100.0, 2)
    tem_superaquecimento = (area_hotspot_px > limiar_area_critica) or (proporcao_haz > 3.5)
    
    # Construção da imagem de diagnóstico com destaque nas zonas críticas
    img_escurecida = (img * 0.45).astype(np.uint8)
    img_resultado = img_escurecida.copy()
    img_resultado[mascara_limpa > 0] = img[mascara_limpa > 0]

    contornos_haz, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    num_focos_criticos = 0
    maior_foco_px = 0
    for cnt in contornos_haz:
        area_c = cv2.contourArea(cnt)
        if area_c > 25:
            num_focos_criticos += 1
            maior_foco_px = max(maior_foco_px, int(area_c))
            cv2.drawContours(img_resultado, [cnt], -1, (0, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_resultado, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 255), 1)
            cv2.putText(img_resultado, f"HAZ: {int(area_c)}px", (x, max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Estimativa de pico térmico equivalente baseado na intensidade do canal V na máscara
    canal_v = hsv[:, :, 2]
    intensidade_media = float(np.mean(canal_v[mascara_limpa > 0])) if area_hotspot_px > 0 else float(np.mean(canal_v))
    temp_estimada_c = round(180.0 + (intensidade_media / 255.0) * 1150.0, 1)

    if tem_superaquecimento:
        alerta_status = "CRITICO_SOBREAQUECIDO"
        acao_clp = "REDUZIR_APORTE_TERMICO_E_CORRENTE"
        diagnostico_detalhado = (
            f"Expansão anormal da Zona Afetada pelo Calor (HAZ) ocupando {proporcao_haz}% do quadro "
            f"({num_focos_criticos} focos ativos | Pico estimado: ~{temp_estimada_c} °C). Alto risco de trinca térmica e distorção metalúrgica."
        )
        causa_raiz = (
            "Aporte térmico (Heat Input) excessivo causado por corrente de soldagem muito elevada, "
            "velocidade de avanço lenta da tocha ou falha na dissipação térmica do dispositivo de fixação."
        )
        plano_correcao = (
            "• Comando imediato ao CLP: Reduzir corrente da fonte de solda em 15% a 20%.\n"
            "• Aumentar velocidade de deslocamento linear do robô em 10% para diminuir a energia por unidade de comprimento (kJ/mm).\n"
            "• Verificar circulação de fluido refrigerante na tocha e nos barramentos de cobre do gabarito."
        )
    else:
        alerta_status = "NORMAL_ESTAVEL"
        acao_clp = "PARAMETROS_TERMICOS_OTIMIZADOS"
        diagnostico_detalhado = (
            f"Perfil térmico contido e simétrico ({proporcao_haz}% de área quente). "
            f"Dissipação de calor adequada na junta soldada sem crescimento excessivo de grão na ZTA."
        )
        causa_raiz = "Balanço energético entre potência do arco elétrico e velocidade de resfriamento operando dentro da especificação metalúrgica."
        plano_correcao = (
            "• Manter parâmetros atuais de corrente e tensão no controlador.\n"
            "• Prosseguir com o ciclo produtivo sem necessidade de resfriamento forçado adicional."
        )

    # HUD Informativo na imagem termográfica
    cor_hud = (0, 50, 255) if tem_superaquecimento else (0, 230, 118)
    cv2.rectangle(img_resultado, (10, 10), (430, 78), (15, 23, 42), -1)
    cv2.rectangle(img_resultado, (10, 10), (430, 78), cor_hud, 1)
    cv2.putText(img_resultado, f"HAZ: {alerta_status}", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_hud, 2)
    cv2.putText(img_resultado, f"Hotspot Area: {area_hotspot_px} px ({proporcao_haz}%)", (20, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 250, 252), 1)
    cv2.putText(img_resultado, f"Focos Criticos: {num_focos_criticos} | Pico Est.: ~{temp_estimada_c} C", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1)

    relatorio_haz = {
        "tipo_inspecao": "Termografia Infravermelha HAZ (Espaço HSV)",
        "alerta_termico_haz": alerta_status,
        "area_ponto_quente_px": area_hotspot_px,
        "proporcao_critica_pct": proporcao_haz,
        "focos_criticos_detectados": num_focos_criticos,
        "temperatura_pico_estimada_c": temp_estimada_c,
        "acao_prescritiva": acao_clp,
        "diagnostico_detalhado": diagnostico_detalhado,
        "causa_raiz": causa_raiz,
        "plano_correcao": plano_correcao,
        "imagem_original_b64": codificar_imagem_base64(img),
        "imagem_processada_b64": codificar_imagem_base64(img_resultado)
    }

    return img_resultado, mascara_limpa, relatorio_haz


if __name__ == "__main__":
    print("Módulo de Visão Computacional Avançado (Padrão AeroMotion-Dynamics) carregado com sucesso.")