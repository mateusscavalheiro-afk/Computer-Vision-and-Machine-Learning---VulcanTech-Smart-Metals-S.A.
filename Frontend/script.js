const API_URL = "http://127.0.0.1:8000";

// Verifica o status de conexão com a API
async function checarStatusAPI() {
    const dot = document.getElementById("apiDot");
    const txt = document.getElementById("apiStatusText");
    try {
        const res = await fetch(`${API_URL}/health`);
        if (res.ok) {
            const data = await res.json();
            dot.className = "dot online";
            txt.innerText = "API Online (Modelos Ativos)";
        } else {
            dot.className = "dot offline";
            txt.innerText = "API Degradada";
        }
    } catch (err) {
        dot.className = "dot offline";
        txt.innerText = "API Desconectada (inicie uvicorn)";
    }
}

// Inicializa verificação periódica a cada 5 segundos
setInterval(checarStatusAPI, 5000);
checarStatusAPI();

// Cenários padrão de telemetria
const cenarios = {
    0: { temp: 48.0, vib: 1.8, torque: 110.0, vazao: 28.0, corr: 25.0, pressao: 160.0, ruido: 58.0 },
    1: { temp: 72.0, vib: 6.2, torque: 195.0, vazao: 22.0, corr: 48.0, pressao: 155.0, ruido: 79.0 },
    2: { temp: 84.0, vib: 2.5, torque: 130.0, vazao: 10.0, corr: 32.0, pressao: 150.0, ruido: 62.0 },
    3: { temp: 55.0, vib: 8.5, torque: 170.0, vazao: 26.0, corr: 39.0, pressao: 95.0, ruido: 92.0 }
};

function carregarCenario(tipo) {
    const c = cenarios[tipo];
    document.getElementById("temperatura_fuso_c").value = c.temp;
    document.getElementById("vibracao_triaxial_rms_mms").value = c.vib;
    document.getElementById("torque_fuso_nm").value = c.torque;
    document.getElementById("vazao_refrigerante_lmin").value = c.vazao;
    document.getElementById("corrente_estator_amp").value = c.corr;
    document.getElementById("pressao_hidraulica_bar").value = c.pressao;
    document.getElementById("nivel_ruido_db").value = c.ruido;
}

// Disparo de inferência em tempo real para a API
async function executarPredicao() {
    const payload = {
        id_celula: document.getElementById("id_celula").value,
        temperatura_fuso_c: parseFloat(document.getElementById("temperatura_fuso_c").value),
        vibracao_triaxial_rms_mms: parseFloat(document.getElementById("vibracao_triaxial_rms_mms").value),
        torque_fuso_nm: parseFloat(document.getElementById("torque_fuso_nm").value),
        vazao_refrigerante_lmin: parseFloat(document.getElementById("vazao_refrigerante_lmin").value),
        corrente_estator_amp: parseFloat(document.getElementById("corrente_estator_amp").value),
        pressao_hidraulica_bar: parseFloat(document.getElementById("pressao_hidraulica_bar").value),
        nivel_ruido_db: parseFloat(document.getElementById("nivel_ruido_db").value)
    };

    document.getElementById("txtCelulaAtiva").innerText = payload.id_celula;

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Erro na resposta da API");
        const data = await res.json();

        // Atualiza Diagnóstico e Ação
        document.getElementById("txtDiagnostico").innerText = data.diagnostico;
        document.getElementById("txtProbabilidade").innerText = `Confiança: ${data.probabilidade_pct}%`;
        
        const box = document.getElementById("statusBox");
        const badge = document.getElementById("badgeAcao");

        box.className = "status-box";
        badge.className = "action-alert";

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

        badge.innerText = `AÇÃO CLP: ${data.acao_recomendada}`;

        // Atualiza OEE
        document.getElementById("oeeDisp").innerText = `${data.impacto_oee.disponibilidade_pct}%`;
        document.getElementById("oeePerf").innerText = `${data.impacto_oee.performance_pct}%`;
        document.getElementById("oeeQual").innerText = `${data.impacto_oee.qualidade_pct}%`;
        
        const totalElem = document.getElementById("oeeTotal");
        totalElem.innerText = `${data.impacto_oee.oee_total_pct}%`;
        totalElem.style.color = data.impacto_oee.oee_total_pct >= 80 ? "var(--success)" : "var(--danger)";

    } catch (err) {
        alert("Falha ao comunicar com a API FastAPI. Verifique se o uvicorn está rodando!");
    }
}

// Inspeção de cordão de solda via OpenCV
async function enviarInspecaoSolda(input) {
    if (!input.files || input.files.length === 0) return;
    const formData = new FormData();
    formData.append("file", input.files[0]);

    const out = document.getElementById("visionOutput");
    out.innerText = "Processando geometria do cordão com Canny e contornos...";

    try {
        const res = await fetch(`${API_URL}/vision/inspect-weld`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        out.innerText = `[SOLDA GEOMÉTRICA]:\nStatus: ${data.status_aprovacao}\nPoros Detectados: ${data.total_porosidades_detectadas}\nÍndice Porosidade: ${data.indice_porosidade_pct}%\nAção: ${data.acao_recomendada}`;
    } catch (err) {
        out.innerText = "Erro ao enviar imagem de solda para a API.";
    }
}

// Análise termográfica HAZ via OpenCV (HSV)
// Inspeção de cordão de solda via OpenCV
async function enviarInspecaoSolda(input) {
    if (!input.files || input.files.length === 0) return;
    const formData = new FormData();
    formData.append("file", input.files[0]);

    const out = document.getElementById("visionOutput");
    out.innerText = "Processando geometria do cordão com Canny e contornos...";

    try {
        const res = await fetch(`${API_URL}/vision/inspect-weld`, {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error("Erro no processamento da imagem de solda.");
        const data = await res.json();
        
        out.innerHTML = `<strong>[INSPEÇÃO GEOMÉTRICA DE SOLDA]:</strong>\n` +
                        `• Status: <strong>${data.status_aprovacao}</strong>\n` +
                        `• Porosidades Detectadas: ${data.total_porosidades_detectadas}\n` +
                        `• Índice de Porosidade: ${data.indice_porosidade_pct}%\n` +
                        `• Ação Recomendada: ${data.acao_recomendada}`;
    } catch (err) {
        out.innerText = "Erro ao enviar imagem de solda para a API. Verifique o console.";
        console.error(err);
    }
}

// Análise termográfica HAZ via OpenCV (HSV)
async function enviarInspecaoHaz(input) {
    if (!input.files || input.files.length === 0) return;
    const formData = new FormData();
    formData.append("file", input.files[0]);

    const out = document.getElementById("visionOutput");
    out.innerText = "Processando análise termográfica HAZ em HSV...";

    try {
        const res = await fetch(`${API_URL}/vision/inspect-haz`, {
            method: "POST",
            body: formData
        });
        if (!res.ok) throw new Error("Erro no processamento termográfico.");
        const data = await res.json();

        out.innerHTML = `<strong>[ANÁLISE TERMOGRÁFICA HAZ]:</strong>\n` +
                        `• Alerta Térmico: <strong>${data.alerta_termico_haz}</strong>\n` +
                        `• Área Hotspot: ${data.area_ponto_quente_px} px\n` +
                        `• Proporção Crítica: ${data.proporcao_critica_pct}%\n` +
                        `• Ação Prescritiva: ${data.acao_prescritiva}`;
    } catch (err) {
        out.innerText = "Erro ao enviar termografia para a API. Verifique o console.";
        console.error(err);
    }
}