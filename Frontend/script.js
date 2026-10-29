const API_URL = "http://127.0.0.1:8000";
let modoVisaoAtual = "solda"; // 'solda' ou 'haz'

// =====================================================================
// 1. NAVEGAÇÃO ENTRE ABAS E MONITORAMENTO DA API
// =====================================================================
function alternarAba(idAba, btnClicado) {
    document.querySelectorAll(".tab-content").forEach(sec => sec.classList.remove("active"));
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.getElementById(idAba).classList.add("active");
    btnClicado.classList.add("active");
}

function atualizarCabecalhoCelula() {
    const cel = document.getElementById("id_celula").value;
    document.getElementById("txtCelulaAtiva").innerText = cel;
}

function registrarLog(mensagem) {
    const term = document.getElementById("terminalLog");
    const agora = new Date().toLocaleTimeString();
    const linha = document.createElement("div");
    linha.innerText = `[${agora}] ${mensagem}`;
    term.prepend(linha);
}

function limparLogTerminal() {
    document.getElementById("terminalLog").innerHTML = "<div>[SISTEMA] Log limpo pelo operador.</div>";
}

async function checarStatusAPI() {
    const dot = document.getElementById("apiDot");
    const txt = document.getElementById("apiStatusText");
    try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
            dot.className = "dot online";
            txt.innerText = "API Online • Modelos Ativos";
        } else {
            dot.className = "dot offline";
            txt.innerText = "API Degradada";
        }
    } catch (err) {
        dot.className = "dot offline";
        txt.innerText = "API Offline (Inicie Uvicorn)";
    }
}

setInterval(checarStatusAPI, 5000);
checarStatusAPI();

// =====================================================================
// 2. BARRAS DINÂMICAS DE SENSORES, CÁLCULO DE IEM E CENÁRIOS
// =====================================================================
const cenariosPadrao = {
    0: { temp: 48.0, vib: 1.8, torque: 110.0, vazao: 28.0, corr: 25.0, pressao: 160.0, ruido: 58.0 },
    1: { temp: 72.0, vib: 6.2, torque: 195.0, vazao: 22.0, corr: 48.0, pressao: 155.0, ruido: 79.0 },
    2: { temp: 88.0, vib: 2.5, torque: 130.0, vazao: 9.5,  corr: 32.0, pressao: 150.0, ruido: 62.0 },
    3: { temp: 55.0, vib: 8.5, torque: 175.0, vazao: 26.0, corr: 39.0, pressao: 92.0,  ruido: 94.0 }
};

function carregarCenario(tipo) {
    const c = cenariosPadrao[tipo];
    document.getElementById("temperatura_fuso_c").value = c.temp;
    document.getElementById("vibracao_triaxial_rms_mms").value = c.vib;
    document.getElementById("torque_fuso_nm").value = c.torque;
    document.getElementById("vazao_refrigerante_lmin").value = c.vazao;
    document.getElementById("corrente_estator_amp").value = c.corr;
    document.getElementById("pressao_hidraulica_bar").value = c.pressao;
    document.getElementById("nivel_ruido_db").value = c.ruido;
    atualizarBarrasSensores();
    registrarLog(`Cenário Simulado [${tipo}] carregado nos sensores IoT.`);
    executarPredicao();
}

function atualizarBarraIndividual(idBarra, valor, minNom, maxNom, maxEscala, inverso = false) {
    const el = document.getElementById(idBarra);
    const pct = Math.min(100, Math.max(5, (valor / maxEscala) * 100));
    el.style.width = `${pct}%`;

    if (inverso) {
        el.style.backgroundColor = valor < minNom ? "var(--status-crit)" : "var(--status-ok)";
    } else {
        if (valor > maxNom || valor < minNom) {
            el.style.backgroundColor = "var(--status-crit)";
        } else if (valor > maxNom * 0.9) {
            el.style.backgroundColor = "var(--status-warn)";
        } else {
            el.style.backgroundColor = "var(--status-ok)";
        }
    }
}

function atualizarBarrasSensores() {
    const temp = parseFloat(document.getElementById("temperatura_fuso_c").value) || 0;
    const vib = parseFloat(document.getElementById("vibracao_triaxial_rms_mms").value) || 0;
    const torque = parseFloat(document.getElementById("torque_fuso_nm").value) || 0;
    const vazao = parseFloat(document.getElementById("vazao_refrigerante_lmin").value) || 0;
    const corr = parseFloat(document.getElementById("corrente_estator_amp").value) || 0;
    const pressao = parseFloat(document.getElementById("pressao_hidraulica_bar").value) || 0;
    const ruido = parseFloat(document.getElementById("nivel_ruido_db").value) || 0;

    atualizarBarraIndividual("bar_temp", temp, 30, 75, 110);
    atualizarBarraIndividual("bar_vib", vib, 0.8, 3.8, 10);
    atualizarBarraIndividual("bar_torque", torque, 45, 210, 250);
    atualizarBarraIndividual("bar_vazao", vazao, 15, 40, 45, true);
    atualizarBarraIndividual("bar_corr", corr, 12, 55, 70);
    atualizarBarraIndividual("bar_pressao", pressao, 110, 220, 250);
    atualizarBarraIndividual("bar_ruido", ruido, 45, 85, 115);

    const iem = (torque * vib) / (vazao + 1.0);
    document.getElementById("topIemVal").innerText = iem.toFixed(2);
}

atualizarBarrasSensores();

// =====================================================================
// 3. INFERÊNCIA PREDITIVA E LAUDO DE ENGENHARIA (POST /predict)
// =====================================================================
async function executarPredicao() {
    const payload = {
        id_celula: document.getElementById("id_celula").value,
        liga_metalica: document.getElementById("liga_metalica").value,
        temperatura_fuso_c: parseFloat(document.getElementById("temperatura_fuso_c").value),
        vibracao_triaxial_rms_mms: parseFloat(document.getElementById("vibracao_triaxial_rms_mms").value),
        torque_fuso_nm: parseFloat(document.getElementById("torque_fuso_nm").value),
        vazao_refrigerante_lmin: parseFloat(document.getElementById("vazao_refrigerante_lmin").value),
        corrente_estator_amp: parseFloat(document.getElementById("corrente_estator_amp").value),
        pressao_hidraulica_bar: parseFloat(document.getElementById("pressao_hidraulica_bar").value),
        nivel_ruido_db: parseFloat(document.getElementById("nivel_ruido_db").value)
    };

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Falha na resposta da API");
        const data = await res.json();

        // Atualiza Banner Principal
        document.getElementById("txtDiagnostico").innerText = data.diagnostico;
        document.getElementById("txtProbabilidade").innerHTML = `Confiança do Modelo Preditivo: <strong>${data.probabilidade_pct}%</strong>`;
        
        const box = document.getElementById("statusBox");
        const badge = document.getElementById("badgeAcao");
        box.className = "status-banner";
        badge.className = "plc-action";

        if (data.tipo_falha === 0) {
            box.classList.add("normal");
            badge.classList.add("normal");
        } else if (data.tipo_falha === 1) {
            box.classList.add("desgaste");
            badge.classList.add("warning");
        } else if (data.tipo_falha === 2) {
            box.classList.add("refrigeracao");
            badge.classList.add("warning");
        } else {
            box.classList.add("trinca");
            badge.classList.add("danger");
        }

        badge.innerText = data.acao_recomendada;

        // Atualiza Análise de Engenharia Completa
        if (data.analise_engenharia) {
            const eng = data.analise_engenharia;
            document.getElementById("txtSeveridade").innerText = `STATUS: ${eng.severidade}`;
            document.getElementById("txtTagClp").innerText = eng.comando_clp_detalhado;
            document.getElementById("txtOQueEstaErrado").innerText = eng.o_que_esta_errado;
            document.getElementById("txtCausaRaiz").innerText = eng.causa_raiz_provavel;

            const containerSintomas = document.getElementById("listaSintomas");
            containerSintomas.innerHTML = "";
            eng.sintomas_detectados.forEach(s => {
                const span = document.createElement("span");
                span.className = data.tipo_falha === 0 ? "symptom-tag ok" : "symptom-tag";
                span.innerText = s;
                containerSintomas.appendChild(span);
            });

            const ulReparo = document.getElementById("listaComoConsertar");
            ulReparo.innerHTML = "";
            eng.como_consertar.forEach(passo => {
                const li = document.createElement("li");
                li.innerText = passo;
                ulReparo.appendChild(li);
            });
        }

        // Atualiza OEE e Barras
        const oee = data.impacto_oee;
        document.getElementById("oeeDisp").innerText = `${oee.disponibilidade_pct}%`;
        document.getElementById("oeePerf").innerText = `${oee.performance_pct}%`;
        document.getElementById("oeeQual").innerText = `${oee.qualidade_pct}%`;
        document.getElementById("oeeTotal").innerText = `${oee.oee_total_pct}%`;

        document.getElementById("barOeeDisp").style.width = `${oee.disponibilidade_pct}%`;
        document.getElementById("barOeePerf").style.width = `${oee.performance_pct}%`;
        document.getElementById("barOeeQual").style.width = `${oee.qualidade_pct}%`;
        document.getElementById("barOeeTotal").style.width = `${oee.oee_total_pct}%`;

        const totalElem = document.getElementById("oeeTotal");
        totalElem.style.color = oee.oee_total_pct >= 80 ? "var(--status-ok)" : "var(--status-crit)";

        registrarLog(`[${payload.id_celula}] Diagnóstico: ${data.diagnostico} -> CLP: ${data.acao_recomendada} (OEE: ${oee.oee_total_pct}%)`);

    } catch (err) {
        registrarLog("[ERRO] Falha ao comunicar com a API FastAPI. Verifique se o Uvicorn está ativo na porta 8000.");
    }
}

// =====================================================================
// 4. LABORATÓRIO DE VISÃO COMPUTACIONAL (COMPARAÇÃO LADO A LADO)
// =====================================================================
function selecionarModoVisao(modo) {
    modoVisaoAtual = modo;
    document.getElementById("btnModoSolda").classList.toggle("active", modo === "solda");
    document.getElementById("btnModoHaz").classList.toggle("active", modo === "haz");

    if (modo === "solda") {
        document.getElementById("lblMetrica1").innerText = "Status do Lote";
        document.getElementById("lblMetrica2").innerText = "Porosidades Detectadas";
        document.getElementById("lblMetrica3").innerText = "Variação de Largura (Volume)";
        document.getElementById("lblMetrica4").innerText = "Desvio Linearidade (Ondulação)";
    } else {
        document.getElementById("lblMetrica1").innerText = "Alerta Térmico HAZ";
        document.getElementById("lblMetrica2").innerText = "Focos Quentes Críticos";
        document.getElementById("lblMetrica3").innerText = "Proporção Crítica HAZ (%)";
        document.getElementById("lblMetrica4").innerText = "Pico Térmico Estimado";
    }
}

async function processarUploadVisao(input) {
    if (!input.files || input.files.length === 0) return;
    await enviarArquivoParaVisao(input.files[0]);
}

async function enviarArquivoParaVisao(arquivoBlob) {
    const formData = new FormData();
    formData.append("file", arquivoBlob, "inspecao.png");

    const endpoint = modoVisaoAtual === "solda" ? "/vision/inspect-weld" : "/vision/inspect-haz";
    const badgeVisao = document.getElementById("badgeStatusVisao");
    badgeVisao.innerText = "PROCESSANDO OPENCV...";

    try {
        const res = await fetch(`${API_URL}${endpoint}`, {
            method: "POST",
            body: formData
        });

        if (!res.ok) throw new Error("Erro no processamento da imagem na API.");
        const data = await res.json();

        // Exibe as imagens lado a lado
        const imgOrig = document.getElementById("imgVisaoOriginal");
        const imgProc = document.getElementById("imgVisaoProcessada");
        document.getElementById("placeholderOriginal").style.display = "none";
        document.getElementById("placeholderProcessada").style.display = "none";

        imgOrig.src = data.imagem_original_b64;
        imgProc.src = data.imagem_processada_b64;
        imgOrig.style.display = "block";
        imgProc.style.display = "block";

        document.getElementById("txtTipoInspecao").innerText = data.tipo_inspecao;

        if (modoVisaoAtual === "solda") {
            const aprovado = data.status_aprovacao === "APROVADO";
            badgeVisao.innerText = data.status_aprovacao;
            badgeVisao.className = `viewport-tag ${aprovado ? "aprovado" : "reprovado"}`;

            document.getElementById("valMetrica1").innerText = data.status_aprovacao;
            document.getElementById("valMetrica1").style.color = aprovado ? "var(--status-ok)" : "var(--status-crit)";
            document.getElementById("valMetrica2").innerText = `${data.total_porosidades_detectadas} poros (${data.indice_porosidade_pct}%)`;
            document.getElementById("valMetrica3").innerText = `${data.variacao_largura_pct}%`;
            document.getElementById("valMetrica4").innerText = `${data.desvio_linearidade_pct}%`;

            document.getElementById("txtClassificacaoGeometria").innerText = data.classificacao_geometria;
            document.getElementById("badgeAcaoVisao").innerText = data.acao_recomendada;
        } else {
            const estavel = data.alerta_termico_haz === "NORMAL_ESTAVEL";
            badgeVisao.innerText = data.alerta_termico_haz;
            badgeVisao.className = `viewport-tag ${estavel ? "aprovado" : "reprovado"}`;

            document.getElementById("valMetrica1").innerText = data.alerta_termico_haz;
            document.getElementById("valMetrica1").style.color = estavel ? "var(--status-ok)" : "var(--status-crit)";
            document.getElementById("valMetrica2").innerText = `${data.focos_criticos_detectados} focos (${data.area_ponto_quente_px} px)`;
            document.getElementById("valMetrica3").innerText = `${data.proporcao_critica_pct}%`;
            document.getElementById("valMetrica4").innerText = `~${data.temperatura_pico_estimada_c} °C`;

            document.getElementById("txtClassificacaoGeometria").innerText = estavel
                ? "DISTRIBUIÇÃO TÉRMICA HOMOGÊNEA E CONTROLADA"
                : "SOBREAQUECIMENTO CRÍTICO NA ZONA AFETADA PELO CALOR (HAZ)";
            document.getElementById("badgeAcaoVisao").innerText = data.acao_prescritiva;
        }

        document.getElementById("txtVisaoDiagnostico").innerText = data.diagnostico_detalhado;
        document.getElementById("txtVisaoCausa").innerText = data.causa_raiz;
        document.getElementById("txtVisaoCorrecao").innerText = data.plano_correcao;

        registrarLog(`[VISÃO 4.0] Inspeção concluída (${data.tipo_inspecao}).`);

    } catch (err) {
        alert("Erro ao processar imagem na API FastAPI. Verifique se o servidor Uvicorn está rodando!");
        console.error(err);
    }
}

// =====================================================================
// 5. GERADOR DE AMOSTRAS SINTÉTICAS REALISTAS (SOLDA EM FUNDO PRETO & HAZ)
// =====================================================================
function gerarAmostraSintetica(qualidade) {
    const canvas = document.getElementById("canvasSintetico");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    // Fundo totalmente preto conforme especificado
    ctx.fillStyle = "#050507";
    ctx.fillRect(0, 0, W, H);

    if (modoVisaoAtual === "solda") {
        // Desenha cordão metálico sobre fundo preto
        const numPassos = 110;
        for (let i = 0; i < numPassos; i++) {
            const x = 40 + (i / numPassos) * (W - 80);
            let yCentro = H / 2;
            let raio = 36;

            if (qualidade === "ruim") {
                // Solda ruim: muito ondulada e com volume irregular
                yCentro += Math.sin(i * 0.22) * 22 + Math.cos(i * 0.45) * 8;
                raio = 36 + Math.sin(i * 0.18) * 18 + (i % 7 === 0 ? 12 : 0);
            } else {
                // Solda boa: reta e uniforme
                yCentro += Math.sin(i * 0.1) * 1.2;
                raio = 35 + Math.sin(i * 0.3) * 1.0;
            }

            const grad = ctx.createRadialGradient(x - 4, yCentro - 5, 3, x, yCentro, raio);
            grad.addColorStop(0, "#e2e8f0");
            grad.addColorStop(0.55, "#94a3b8");
            grad.addColorStop(0.9, "#475569");
            grad.addColorStop(1, "#090d16");

            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(x, yCentro, raio, 0, Math.PI * 2);
            ctx.fill();
        }

        // Se for amostra ruim, injeta porosidades escuras internas no cordão
        if (qualidade === "ruim") {
            const poros = [
                { x: 160, y: H / 2 + 12, r: 5 },
                { x: 220, y: H / 2 - 14, r: 6 },
                { x: 295, y: H / 2 + 5, r: 4 },
                { x: 350, y: H / 2 - 18, r: 6 },
                { x: 420, y: H / 2 + 15, r: 5 },
                { x: 485, y: H / 2 - 6, r: 6 },
                { x: 530, y: H / 2 + 10, r: 4 }
            ];
            poros.forEach(p => {
                ctx.fillStyle = "#030305";
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = "#1e293b";
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }
    } else {
        // Simulação Termográfica HAZ (Espaço HSV)
        ctx.fillStyle = "#0a1128";
        ctx.fillRect(0, 0, W, H);

        // Faixa térmica base do cordão
        const gradBase = ctx.createLinearGradient(0, H / 2 - 70, 0, H / 2 + 70);
        gradBase.addColorStop(0, "#0a1128");
        gradBase.addColorStop(0.3, "#4c1d95");
        gradBase.addColorStop(0.5, qualidade === "boa" ? "#ea580c" : "#ff0000");
        gradBase.addColorStop(0.7, "#4c1d95");
        gradBase.addColorStop(1, "#0a1128");

        ctx.fillStyle = gradBase;
        ctx.fillRect(40, H / 2 - 65, W - 80, 130);

        if (qualidade === "ruim") {
            // Grandes manchas de sobreaquecimento crítico (Vermelho puro / Amarelo quente)
            const hotspots = [
                { x: 200, y: H / 2, r: 58 },
                { x: 340, y: H / 2 - 10, r: 65 },
                { x: 460, y: H / 2 + 8, r: 52 }
            ];
            hotspots.forEach(h => {
                const gHot = ctx.createRadialGradient(h.x, h.y, 5, h.x, h.y, h.r);
                gHot.addColorStop(0, "#ffff00");
                gHot.addColorStop(0.45, "#ff0000");
                gHot.addColorStop(0.85, "#ea580c");
                gHot.addColorStop(1, "transparent");
                ctx.fillStyle = gHot;
                ctx.beginPath();
                ctx.arc(h.x, h.y, h.r, 0, Math.PI * 2);
                ctx.fill();
            });
        }
    }

    canvas.toBlob(blob => {
        if (blob) enviarArquivoParaVisao(blob);
    }, "image/png");
}