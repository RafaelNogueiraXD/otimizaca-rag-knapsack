"""Painel do experimento: compara os seletores de contexto nos resultados de `uv run task avaliar`.

uv run task painel
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from rag_knapsack.config import RAIZ
from rag_knapsack.dados import Chunk
from rag_knapsack.evals import resumir_linhas
from rag_knapsack.metricas import normalizar_resposta
from rag_knapsack.seletores import ALGORITMOS, criar_seletor

PASTA_RESULTADOS = Path(os.getenv("PAINEL_RESULTADOS", str(RAIZ / "resultados")))

NOMES = {"topk": "Top-K", "greedy": "Greedy", "knapsack": "Knapsack DP", "topk_corte": "Top-K com corte"}
DESCRICOES = {
    "topk": "os k chunks de maior score, sem olhar o tamanho (pode estourar T)",
    "greedy": "maior score por token primeiro, pulando o que não cabe em T",
    "knapsack": "mochila 0/1 exata: maior soma de scores que cabe em T",
    "topk_corte": "maior score primeiro até encher T, sem limite de k (base justa para Greedy e Knapsack)",
}

# Paleta categórica validada para daltonismo nos dois modos (ordem dos slots = ordem de ALGORITMOS).
# A cor acompanha o algoritmo: filtrar algoritmos não repinta os que sobram.
CORES = {
    "light": {"topk": "#2a78d6", "greedy": "#eb6834", "knapsack": "#1baf7a", "topk_corte": "#eda100"},
    "dark": {"topk": "#3987e5", "greedy": "#d95926", "knapsack": "#199e70", "topk_corte": "#c98500"},
}
TINTA = {
    "light": {
        "texto": "#0b0b0b",
        "secundario": "#52514e",
        "mudo": "#898781",
        "grade": "#e1e0d9",
        "eixo": "#c3c2b7",
        "superficie": "#fcfcfb",
    },
    "dark": {
        "texto": "#ffffff",
        "secundario": "#c3c2b7",
        "mudo": "#898781",
        "grade": "#2c2c2a",
        "eixo": "#383835",
        "superficie": "#1a1a19",
    },
}

ROTULOS = {
    "acuracia_juiz": "Acurácia (juiz)",
    "f1": "F1",
    "em": "EM",
    "taxa_sem_resposta": "Respondeu 'unknown'",
    "recall_suporte": "Recall de suporte",
    "suporte_completo": "Os 2 parágrafos de suporte",
    "precisao_suporte": "Precisão de suporte",
    "taxa_estouro": "Estourou T",
    "ocupacao": "Ocupação de T",
    "atingiu_otimo": "Atingiu o ótimo",
    "razao_otimo": "Valor ÷ ótimo",
    "valor_total": "Valor total (soma dos scores)",
    "tokens_usados": "Tokens de contexto",
    "custo_medio_usd": "Custo por pergunta (US$)",
    "erros": "Erros de API",
}
FRACOES = {
    "acuracia_juiz",
    "f1",
    "em",
    "taxa_sem_resposta",
    "recall_suporte",
    "suporte_completo",
    "precisao_suporte",
    "taxa_estouro",
    "ocupacao",
    "atingiu_otimo",
    "razao_otimo",
}

# Exemplo da seção 8 da proposta ("Onde fica Brasília?"): (id, resumo, tokens, score).
BRASILIA = [
    ("c1", "Parágrafo de abertura inteiro: capital federal, 1960, Lúcio Costa e Niemeyer, UNESCO...", 280, 0.93),
    ("c2", "Brasília é a capital do Brasil e sede do Distrito Federal, no Centro-Oeste.", 160, 0.88),
    ("c3", "O Distrito Federal fica em Goiás, no Planalto Central, a 15°47′S 47°52′O.", 140, 0.80),
    ("c4", "A altitude média é de 1.172 m; faz divisa com Luziânia, Formosa e Padre Bernardo.", 110, 0.70),
    ("c5", "Inaugurada em 21 de abril de 1960, no governo de Juscelino Kubitschek.", 90, 0.41),
    ("c6", "Ver também: Plano Piloto · Eixo Monumental · Lista de capitais.", 60, 0.22),
]


# ---------------------------------------------------------------------------------------
# Aparência dos gráficos


def num(valor: float, casas: int = 2) -> str:
    """Número no formato brasileiro (1.234,56)."""
    return f"{valor:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def tema_atual() -> str:
    tipo = getattr(st.context.theme, "type", None)
    return "dark" if tipo == "dark" else "light"


def escala_cor(algoritmos: list[str], tema: str, legenda: bool = True) -> alt.Color:
    """Quando o eixo Y já nomeia cada algoritmo, a legenda seria redundante."""
    return alt.Color(
        "Algoritmo:N",
        scale=alt.Scale(domain=[NOMES[a] for a in algoritmos], range=[CORES[tema][a] for a in algoritmos]),
        legend=alt.Legend(title=None, orient="top", symbolType="circle", columns=4, labelLimit=160)
        if legenda
        else None,
    )


def configurar(grafico: alt.TopLevelMixin, tema: str) -> alt.TopLevelMixin:
    tinta = TINTA[tema]
    # Sem o padding à esquerda, o Streamlit corta o fim dos rótulos longos do eixo Y.
    return (
        grafico.properties(padding={"left": 20, "right": 8, "top": 8, "bottom": 4})
        .configure(background="transparent", font="sans-serif")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=tinta["mudo"],
            titleColor=tinta["secundario"],
            gridColor=tinta["grade"],
            domainColor=tinta["eixo"],
            tickColor=tinta["eixo"],
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
        )
        .configure_legend(labelColor=tinta["secundario"], labelFontSize=12)
        .configure_title(color=tinta["texto"], anchor="start", fontSize=14, fontWeight="normal")
        .configure_header(labelColor=tinta["secundario"], titleColor=tinta["secundario"], labelFontSize=12)
    )


def mostrar(grafico: alt.TopLevelMixin) -> None:
    st.altair_chart(grafico, width="stretch", theme=None)


# ---------------------------------------------------------------------------------------
# Dados


def listar_execucoes(pasta: Path) -> list[Path]:
    if not pasta.exists():
        return []
    return sorted((p for p in pasta.glob("avaliacao-*") if (p / "linhas.jsonl").exists()), reverse=True)


def _jsonl(caminho: Path) -> list[dict]:
    with caminho.open(encoding="utf-8") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


@st.cache_data(show_spinner=False)
def carregar_execucao(pasta: str, marca: float) -> tuple[list[dict], dict[str, dict], list[dict], dict]:
    caminho = Path(pasta)
    linhas = _jsonl(caminho / "linhas.jsonl")
    perguntas = {registro["id"]: registro for registro in _jsonl(caminho / "perguntas.jsonl")}
    desempenho = json.loads((caminho / "desempenho.json").read_text(encoding="utf-8"))
    execucao = json.loads((caminho / "execucao.json").read_text(encoding="utf-8"))
    return linhas, perguntas, desempenho, execucao


def _metadados(pasta: Path) -> dict:
    try:
        return json.loads((pasta / "execucao.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _rotulo_execucao(pasta: Path, metadados: dict) -> str:
    data = pasta.name.removeprefix("avaliacao-")
    detalhes = [f"{metadados['perguntas']} perguntas"] if "perguntas" in metadados else []
    detalhes.append(metadados.get("modelo_gerador") or "sem LLM")
    return f"{data} · {' · '.join(detalhes)}"


def tabela_resumo(linhas: list[dict], algoritmos: list[str], orcamentos: list[int]) -> pd.DataFrame:
    registros = []
    for orcamento in orcamentos:
        for algoritmo in algoritmos:
            grupo = [linha for linha in linhas if linha["algoritmo"] == algoritmo and linha["orcamento"] == orcamento]
            if not grupo:
                continue
            resumo = resumir_linhas(grupo)
            if "custo_total_usd" in resumo:
                resumo["custo_medio_usd"] = resumo["custo_total_usd"] / resumo["perguntas"]
            registros.append({"algoritmo": algoritmo, "Algoritmo": NOMES[algoritmo], "orcamento": orcamento, **resumo})
    return pd.DataFrame(registros)


def tabela_formatada(resumos: pd.DataFrame, metricas: list[str]) -> None:
    """A tabela é a visão acessível dos gráficos: todos os valores, sem depender de cor."""
    colunas = ["Algoritmo", *[m for m in metricas if m in resumos]]
    configuracao = {}
    for metrica in colunas[1:]:
        if metrica in FRACOES:
            formato = "percent"
        elif metrica == "custo_medio_usd":
            formato = "%.5f"
        elif metrica in ("tokens_usados", "erros"):
            formato = "%.0f"
        else:
            formato = "%.2f"
        configuracao[metrica] = st.column_config.NumberColumn(ROTULOS.get(metrica, metrica), format=formato)
    st.dataframe(resumos[colunas], hide_index=True, column_config=configuracao)


# ---------------------------------------------------------------------------------------
# Gráficos


def barras_agrupadas(resumos: pd.DataFrame, metricas: list[str], algoritmos: list[str], tema: str, titulo: str):
    metricas = [m for m in metricas if m in resumos and resumos[m].notna().any()]
    dados = resumos.melt(id_vars=["Algoritmo"], value_vars=metricas, var_name="metrica", value_name="valor")
    dados["Métrica"] = dados["metrica"].map(ROTULOS)
    base = alt.Chart(dados).encode(
        y=alt.Y(
            "Métrica:N",
            sort=[ROTULOS[m] for m in metricas],
            title=None,
            scale=alt.Scale(paddingInner=0.3),
            axis=alt.Axis(labelLimit=240, ticks=False, domain=False),
        ),
        yOffset=alt.YOffset("Algoritmo:N", sort=[NOMES[a] for a in algoritmos], scale=alt.Scale(paddingInner=0.12)),
        x=alt.X(
            "valor:Q",
            title=None,
            scale=alt.Scale(domain=[0, 1.12], nice=False),
            axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1]),
        ),
    )
    barras = base.mark_bar(cornerRadiusEnd=4).encode(
        color=escala_cor(algoritmos, tema),
        tooltip=[alt.Tooltip("Algoritmo:N"), alt.Tooltip("Métrica:N"), alt.Tooltip("valor:Q", format=".1%")],
    )
    rotulos = base.mark_text(
        align="left", baseline="middle", dx=4, fontSize=11, color=TINTA[tema]["secundario"]
    ).encode(text=alt.Text("valor:Q", format=".0%"))
    return configurar(alt.layer(barras, rotulos).properties(title=titulo, height=alt.Step(17)), tema)


def barras_tokens(resumos: pd.DataFrame, algoritmos: list[str], orcamento: int, tema: str):
    base = alt.Chart(resumos).encode(
        y=alt.Y(
            "Algoritmo:N",
            sort=[NOMES[a] for a in algoritmos],
            title=None,
            scale=alt.Scale(paddingInner=0.35),
            axis=alt.Axis(ticks=False, domain=False),
        ),
        x=alt.X(
            "tokens_usados:Q",
            title="tokens de contexto por pergunta (média)",
            scale=alt.Scale(domainMax=max(orcamento, resumos["tokens_usados"].max()) * 1.18),
        ),
    )
    barras = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=escala_cor(algoritmos, tema, legenda=False),
        tooltip=[
            alt.Tooltip("Algoritmo:N"),
            alt.Tooltip("tokens_usados:Q", title="tokens", format=".0f"),
            alt.Tooltip("taxa_estouro:Q", title="estourou T em", format=".0%"),
        ],
    )
    rotulos = base.mark_text(align="left", dx=4, fontSize=11, color=TINTA[tema]["secundario"]).encode(
        text=alt.Text("tokens_usados:Q", format=".0f")
    )
    limite = pd.DataFrame({"T": [orcamento], "rotulo": [f"T = {orcamento}"]})
    regra = alt.Chart(limite).mark_rule(color=TINTA[tema]["texto"], strokeWidth=1.5).encode(x="T:Q")
    texto_regra = (
        alt.Chart(limite)
        .mark_text(align="left", dx=4, dy=-8, fontSize=11, color=TINTA[tema]["texto"])
        .encode(x="T:Q", y=alt.value(0), text="rotulo:N")
    )
    return configurar(
        alt.layer(barras, rotulos, regra, texto_regra).properties(title="Tamanho do contexto", height=alt.Step(30)),
        tema,
    )


def linhas_por_orcamento(resumos: pd.DataFrame, metrica: str, algoritmos: list[str], tema: str):
    fracao = metrica in FRACOES
    formato = ".0%" if fracao else {"custo_medio_usd": ".5f", "valor_total": ".2f"}.get(metrica, ".0f")
    grafico = (
        alt.Chart(resumos)
        .mark_line(
            strokeWidth=2,
            point=alt.OverlayMarkDef(size=80, filled=True, stroke=TINTA[tema]["superficie"], strokeWidth=2),
        )
        .encode(
            x=alt.X("orcamento:O", title="orçamento T (tokens)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                f"{metrica}:Q",
                title=ROTULOS[metrica],
                axis=alt.Axis(format=formato),
                scale=alt.Scale(domain=[0, 1]) if fracao else alt.Scale(zero=True),
            ),
            color=escala_cor(algoritmos, tema),
            tooltip=[
                alt.Tooltip("Algoritmo:N"),
                alt.Tooltip("orcamento:O", title="T"),
                alt.Tooltip(f"{metrica}:Q", title=ROTULOS[metrica], format=formato.replace(".0%", ".1%")),
            ],
        )
    )
    return configurar(grafico.properties(height=340), tema)


def faixa_otimo(linhas: list[dict], algoritmos: list[str], orcamento: int, tema: str):
    """Um traço por pergunta: onde cada seleção válida fica em relação ao ótimo da mochila."""
    dados = pd.DataFrame(
        [
            {
                "Algoritmo": NOMES[linha["algoritmo"]],
                "razao": min(linha["razao_otimo"], 1.0),
                "pergunta": linha["pergunta"],
            }
            for linha in linhas
            if linha["orcamento"] == orcamento and linha["algoritmo"] in algoritmos and not linha["estourou_orcamento"]
        ]
    )
    if dados.empty:
        return None
    ordem = [NOMES[a] for a in algoritmos]
    eixo_y = alt.Y("Algoritmo:N", sort=ordem, title=None, axis=alt.Axis(ticks=False, domain=False))
    tracos = (
        alt.Chart(dados)
        .mark_tick(thickness=2, size=22, opacity=0.55)
        .encode(
            y=eixo_y,
            x=alt.X(
                "razao:Q",
                title="valor da seleção ÷ valor ótimo (cada traço é uma pergunta)",
                scale=alt.Scale(domain=[min(0.5, float(dados["razao"].min())), 1.0]),
                axis=alt.Axis(format="%"),
            ),
            color=escala_cor(algoritmos, tema, legenda=False),
            tooltip=[
                alt.Tooltip("Algoritmo:N"),
                alt.Tooltip("razao:Q", title="valor ÷ ótimo", format=".1%"),
                alt.Tooltip("pergunta:N"),
            ],
        )
    )
    media = (
        alt.Chart(dados)
        .mark_point(shape="diamond", filled=True, size=90, color=TINTA[tema]["texto"])
        .encode(y=eixo_y, x="mean(razao):Q", tooltip=[alt.Tooltip("mean(razao):Q", title="média", format=".1%")])
    )
    return configurar(
        alt.layer(tracos, media).properties(title="Distribuição por pergunta (◆ = média)", height=alt.Step(34)), tema
    )


def pontos_latencia(desempenho: pd.DataFrame, algoritmos: list[str], tema: str):
    grafico = (
        alt.Chart(desempenho)
        .mark_point(filled=True, size=110, stroke=TINTA[tema]["superficie"], strokeWidth=2)
        .encode(
            y=alt.Y(
                "Algoritmo:N", sort=[NOMES[a] for a in algoritmos], title=None, axis=alt.Axis(ticks=False, domain=False)
            ),
            x=alt.X("ms_por_selecao:Q", title="ms por seleção (escala log)", scale=alt.Scale(type="log")),
            color=escala_cor(algoritmos, tema, legenda=False),
            tooltip=[
                alt.Tooltip("Algoritmo:N"),
                alt.Tooltip("ms_por_selecao:Q", title="ms (média)", format=".4f"),
                alt.Tooltip("memoria_pico_mib:Q", title="pico de memória (MiB)", format=".3f"),
            ],
        )
    )
    return configurar(grafico.properties(title="Tempo de seleção (PerformanceEval)", height=alt.Step(34)), tema)


# ---------------------------------------------------------------------------------------
# Abas


def aba_visao_geral(resumos: pd.DataFrame, algoritmos: list[str], orcamento: int, tema: str, usou_llm: bool) -> None:
    atual = resumos[resumos["orcamento"] == orcamento]
    if atual.empty:
        st.info("Nenhum resultado para este orçamento com os filtros atuais.")
        return
    st.caption(f"T = {orcamento} tokens · {int(atual['perguntas'].iloc[0])} perguntas por algoritmo")
    erros = int(atual["erros"].sum()) if "erros" in atual else 0
    if erros:
        st.warning(
            f"{erros} respostas com erro de API neste orçamento; elas contam como erradas. Rode o mesmo "
            "`uv run task avaliar` de novo para refazer só as falhas (o resto vem do cache)."
        )
    tem_respostas = "f1" in atual and atual["f1"].notna().any()
    esquerda, direita = st.columns(2, gap="large")
    with esquerda:
        if tem_respostas:
            mostrar(barras_agrupadas(atual, ["acuracia_juiz", "f1", "em"], algoritmos, tema, "Qualidade da resposta"))
        elif usou_llm:
            st.error("Nenhuma resposta do LLM neste orçamento: todas as chamadas falharam.")
        else:
            st.info("Execução sem respostas do LLM (--sem-llm): só as métricas de seleção.")
        mostrar(barras_tokens(atual, algoritmos, orcamento, tema))
    with direita:
        mostrar(
            barras_agrupadas(
                atual,
                ["recall_suporte", "suporte_completo", "precisao_suporte", "taxa_estouro"],
                algoritmos,
                tema,
                "Contexto selecionado",
            )
        )
    st.markdown("**Todas as métricas neste orçamento**")
    tabela_formatada(
        atual,
        [
            "acuracia_juiz",
            "f1",
            "em",
            "taxa_sem_resposta",
            "recall_suporte",
            "suporte_completo",
            "precisao_suporte",
            "taxa_estouro",
            "tokens_usados",
            "ocupacao",
            "valor_total",
            "razao_otimo",
            "atingiu_otimo",
            "custo_medio_usd",
            "erros",
        ],
    )


def aba_orcamento(resumos: pd.DataFrame, algoritmos: list[str], tema: str) -> None:
    candidatas = [
        "acuracia_juiz",
        "f1",
        "em",
        "recall_suporte",
        "suporte_completo",
        "taxa_estouro",
        "tokens_usados",
        "valor_total",
        "razao_otimo",
        "custo_medio_usd",
    ]
    disponiveis = [m for m in candidatas if m in resumos and resumos[m].notna().any()]
    metrica = st.segmented_control(
        "Métrica", disponiveis, format_func=ROTULOS.get, default=disponiveis[0], required=True, key="metrica_orcamento"
    )
    st.caption(
        "Cada ponto é a média das perguntas naquele orçamento; mais tokens de contexto custam mais por chamada. "
        "Linhas sobrepostas são algoritmos com o mesmo valor (a tabela abaixo mostra todos)."
    )
    mostrar(linhas_por_orcamento(resumos, metrica, algoritmos, tema))
    pivo = resumos.pivot(index="Algoritmo", columns="orcamento", values=metrica)
    pivo = pivo.reindex([NOMES[a] for a in algoritmos if NOMES[a] in pivo.index])
    pivo.columns = [f"T = {c}" for c in pivo.columns]
    st.dataframe(pivo.style.format("{:.1%}" if metrica in FRACOES else "{:.4g}"))


def aba_otimalidade(
    linhas: list[dict], resumos: pd.DataFrame, desempenho: list[dict], algoritmos: list[str], orcamento: int, tema: str
) -> None:
    atual = resumos[resumos["orcamento"] == orcamento]
    st.markdown(
        "A mochila maximiza a **soma dos scores** no orçamento T. Aqui se vê quanto cada algoritmo fica do ótimo "
        "dessa soma (H2) e quanto tempo a seleção custa (H3)."
    )
    if not atual.empty and {"knapsack", "topk_corte"} <= set(atual["algoritmo"]):
        dp = atual[atual["algoritmo"] == "knapsack"].iloc[0]
        corte = atual[atual["algoritmo"] == "topk_corte"].iloc[0]
        comparacao = (
            f"Em T = {orcamento}, o Knapsack DP somou {num(dp['valor_total'])} de score contra "
            f"{num(corte['valor_total'])} do Top-K com corte, com recall de suporte de {dp['recall_suporte']:.0%} "
            f"contra {corte['recall_suporte']:.0%}."
        )
        if dp["recall_suporte"] < corte["recall_suporte"]:
            comparacao += (
                " Aqui, valor maior na mochila não trouxe contexto melhor: a soma dos scores favorece "
                "levar mais chunks, mesmo pouco relevantes."
            )
        st.info(comparacao)
    esquerda, direita = st.columns([3, 2], gap="large")
    with esquerda:
        if not atual.empty:
            mostrar(barras_agrupadas(atual, ["atingiu_otimo", "razao_otimo"], algoritmos, tema, "Distância do ótimo"))
        grafico = faixa_otimo(linhas, algoritmos, orcamento, tema)
        if grafico is not None:
            mostrar(grafico)
        estouros = sum(
            1
            for linha in linhas
            if linha["orcamento"] == orcamento and linha["algoritmo"] in algoritmos and linha["estourou_orcamento"]
        )
        if estouros:
            st.caption(
                f"{estouros} seleções estouraram T e ficaram fora da distribuição: não são soluções válidas da mochila."
            )
    with direita:
        perf = pd.DataFrame([d for d in desempenho if d["orcamento"] == orcamento and d["algoritmo"] in algoritmos])
        if not perf.empty:
            perf["Algoritmo"] = perf["algoritmo"].map(NOMES)
            mostrar(pontos_latencia(perf, algoritmos, tema))
            duracoes = [linha["duracao_geracao_s"] for linha in linhas if linha.get("duracao_geracao_s")]
            mais_lento = perf["ms_por_selecao"].max()
            st.metric("Seleção mais lenta (média)", f"{num(mais_lento, 3)} ms")
            if duracoes:
                mediana = float(pd.Series(duracoes).median())
                st.metric(
                    "Geração pelo LLM (mediana)",
                    f"{num(mediana)} s",
                    help="Tempo de uma chamada ao gerador, medido pelo agno.",
                )
                vezes = num(mediana * 1000 / mais_lento, 0)
                st.caption(f"A chamada ao LLM leva cerca de {vezes}× o tempo da seleção mais lenta.")
            st.dataframe(
                perf[["Algoritmo", "ms_por_selecao", "ms_por_selecao_mediana", "memoria_pico_mib"]],
                hide_index=True,
                column_config={
                    "ms_por_selecao": st.column_config.NumberColumn("ms (média)", format="%.4f"),
                    "ms_por_selecao_mediana": st.column_config.NumberColumn("ms (mediana)", format="%.4f"),
                    "memoria_pico_mib": st.column_config.NumberColumn("pico de memória (MiB)", format="%.3f"),
                },
            )


def md(texto: str) -> str:
    """Escapa o texto dos dados para o markdown do Streamlit (um par de '$' viraria fórmula LaTeX)."""
    return re.sub(r"([\\`*_\[\]<>#|~$])", r"\\\1", str(texto))


def _veredito(linha: dict, criterio: str | None) -> str:
    correta = _correta(linha, criterio)
    if correta is None:
        return "Sem veredito"
    rotulo = "Correta" if correta else "Incorreta"
    return f"{'✅' if correta else '❌'} {rotulo}" + (" (EM)" if criterio == "em" else "")


def _correta(linha: dict, criterio: str | None) -> bool | None:
    if criterio == "juiz":
        return linha.get("juiz_correto")
    if criterio == "em" and linha.get("resposta_gerada") is not None:
        return linha.get("em") == 1.0
    return None


def aba_perguntas(linhas: list[dict], perguntas: dict[str, dict], algoritmos: list[str], orcamento: int) -> None:
    por_pergunta: dict[str, dict[str, dict]] = {}
    for linha in linhas:
        if linha["orcamento"] == orcamento and linha["algoritmo"] in algoritmos:
            por_pergunta.setdefault(linha["id"], {})[linha["algoritmo"]] = linha
    todas = [linha for respostas in por_pergunta.values() for linha in respostas.values()]
    # Acerto pelo juiz; sem juiz (--sem-juiz), por EM; sem respostas (--sem-llm), não há acerto.
    if any(linha.get("juiz_correto") is not None for linha in todas):
        criterio = "juiz"
    elif any(linha.get("resposta_gerada") is not None for linha in todas):
        criterio = "em"
    else:
        criterio = None

    filtro1, filtro2, filtro3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    busca = filtro1.text_input("Buscar na pergunta ou no gabarito", key="busca")
    so_divergencias = filtro2.toggle(
        "Só divergências",
        key="so_divergencias",
        help="Perguntas em que os algoritmos escolheram contextos, deram respostas ou tiveram vereditos diferentes",
    )
    so_erros = criterio is not None and filtro3.toggle(
        "Só com erro", key="so_erros", help="Perguntas em que ao menos um algoritmo errou"
    )

    def entra(pid: str) -> bool:
        respostas = por_pergunta[pid]
        primeira = next(iter(respostas.values()))
        texto = f"{primeira['pergunta']} {primeira['resposta_referencia']}".lower()
        if busca and busca.lower() not in texto:
            return False
        if so_divergencias:
            vereditos = {_correta(r, criterio) for r in respostas.values()}
            textos = {normalizar_resposta(r["resposta_gerada"] or "") for r in respostas.values()}
            contextos = {tuple(r["selecionados"]) for r in respostas.values()}
            if len(vereditos) <= 1 and len(textos) <= 1 and len(contextos) <= 1:
                return False
        if so_erros and all(_correta(r, criterio) for r in respostas.values()):
            return False
        return True

    ids = [pid for pid in por_pergunta if entra(pid)]
    if not ids:
        st.info("Nenhuma pergunta com esses filtros.")
        return
    resumo = f"{len(ids)} perguntas"
    if criterio is not None:
        acertos = {a: sum(1 for pid in ids if _correta(por_pergunta[pid].get(a, {}), criterio)) for a in algoritmos}
        fonte = "segundo o juiz" if criterio == "juiz" else "por EM (execução sem juiz)"
        resumo += f" · acertos {fonte}: " + " · ".join(f"{NOMES[a]} {acertos[a]}/{len(ids)}" for a in algoritmos)
    st.caption(resumo)
    por_pagina = 8
    paginas = (len(ids) - 1) // por_pagina + 1
    pagina = 1
    if paginas > 1:
        pagina = st.columns([1, 5])[0].number_input(
            f"Página (de {paginas})", min_value=1, max_value=paginas, value=1, key="pagina"
        )
    for pid in ids[(pagina - 1) * por_pagina : pagina * por_pagina]:
        respostas = por_pergunta[pid]
        primeira = next(iter(respostas.values()))
        registro = perguntas.get(pid, {})
        titulos = {c["id"]: c["titulo"] for c in registro.get("chunks", [])}
        candidatos = {c["id"]: c for c in primeira["candidatos"]}
        suporte = [c for c in primeira["candidatos"] if c["de_suporte"]]
        tokens_suporte = sum(c["tokens"] for c in suporte)
        with st.container(border=True):
            st.markdown(f"**{md(primeira['pergunta'])}**")
            st.markdown(
                f"Resposta ideal: **{md(primeira['resposta_referencia'])}** · Contexto ideal: "
                + ", ".join(f"★ {md(titulos.get(c['id'], c['id']))}" for c in suporte)
                + f" ({tokens_suporte} tokens{', cabe' if tokens_suporte <= orcamento else ', não cabe'} em T)"
            )
            colunas = st.columns(len(algoritmos))
            for coluna, algoritmo in zip(colunas, algoritmos, strict=True):
                linha = respostas.get(algoritmo)
                with coluna:
                    st.markdown(f"**{NOMES[algoritmo]}**")
                    if linha is None:
                        st.caption("sem resultado")
                        continue
                    if linha.get("resposta_gerada") is not None:
                        st.markdown(f"“{md(linha['resposta_gerada'])}”")
                        if criterio is not None:
                            st.markdown(f"{_veredito(linha, criterio)} · F1 {num(linha['f1'])}")
                    if linha.get("erro"):
                        origem = "do juiz" if linha.get("resposta_gerada") is not None else "da geração"
                        st.caption(f"⚠ erro {origem}: {md(linha['erro'])}")
                    estouro = " · ⚠ estourou T" if linha["estourou_orcamento"] else ""
                    st.caption(f"{linha['tokens_usados']} tokens{estouro} · recall {linha['recall_suporte']:.0%}")
                    st.caption(
                        "  \n".join(
                            f"{'★' if candidatos[cid]['de_suporte'] else '·'} {md(titulos.get(cid, cid))}"
                            for cid in linha["selecionados"]
                        )
                    )
            with st.expander("Justificativas do juiz e parágrafos candidatos"):
                for algoritmo in algoritmos:
                    linha = respostas.get(algoritmo)
                    if linha and linha.get("juiz_justificativa"):
                        st.markdown(f"**{NOMES[algoritmo]}**: {md(linha['juiz_justificativa'])}")
                for chunk in sorted(registro.get("chunks", []), key=lambda c: -c["score"]):
                    marca = "★ suporte · " if chunk["de_suporte"] else ""
                    st.markdown(
                        f"**{md(chunk['titulo'])}** ({marca}score {chunk['score']:.3f} · {chunk['tokens']} tokens)  \n"
                        f"{md(chunk['texto'])}"
                    )


def aba_brasilia(tema: str) -> None:
    st.markdown(
        "Exemplo da seção 8 da proposta: **“Onde fica Brasília?”**. Ajuste o orçamento e veja o que cada "
        "algoritmo escolhe. Aqui não há LLM: o valor de cada chunk é o score dado na proposta."
    )
    controle1, controle2 = st.columns(2)
    orcamento = controle1.slider("Orçamento T (tokens)", 60, 840, 300, step=10, key="t_brasilia")
    k = controle2.slider("k do Top-K", 1, 6, 1, key="k_brasilia")
    chunks = [Chunk(id=i, titulo=i, texto=t, tokens=w, score=v) for i, t, w, v in BRASILIA]
    selecoes = {a: criar_seletor(a, k=k).selecionar(chunks, orcamento) for a in ALGORITMOS}
    tabela = pd.DataFrame(
        [
            {
                "chunk": i,
                "conteúdo": t,
                "tokens (w)": w,
                "score (v)": v,
                "v/w ×1000": round(v / w * 1000, 2),
                **{NOMES[a]: "✓" if i in {c.id for c in selecoes[a].chunks} else "" for a in ALGORITMOS},
            }
            for i, t, w, v in BRASILIA
        ]
    )
    # Larguras fixas: numa aba que ainda não foi aberta, a medição automática deixa colunas vazias.
    configuracao = {
        "chunk": st.column_config.TextColumn(width=50),
        "conteúdo": st.column_config.TextColumn(width="large"),
        "tokens (w)": st.column_config.NumberColumn(width=80),
        "score (v)": st.column_config.NumberColumn(width=60),
        "v/w ×1000": st.column_config.NumberColumn(width=80),
        **{NOMES[a]: st.column_config.TextColumn(width=95) for a in ALGORITMOS},
    }
    st.dataframe(tabela, hide_index=True, column_config=configuracao)
    colunas = st.columns(len(ALGORITMOS))
    otimo = selecoes["knapsack"].valor_total
    for coluna, algoritmo in zip(colunas, ALGORITMOS, strict=True):
        selecao = selecoes[algoritmo]
        aviso = " · ⚠ estourou T" if selecao.estourou_orcamento else ""
        coluna.metric(NOMES[algoritmo], num(selecao.valor_total), help=DESCRICOES[algoritmo], border=True)
        coluna.caption(
            f"{', '.join(c.id for c in selecao.chunks) or 'nada'} · {selecao.tokens_usados}/{orcamento} tokens{aviso}"
            + ("" if selecao.estourou_orcamento or otimo == 0 else f" · {selecao.valor_total / otimo:.0%} do ótimo")
        )


# ---------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="RAG com mochila", page_icon="🎒", layout="wide")
    st.title("Seleção de contexto em RAG como Problema da Mochila")
    tema = tema_atual()

    execucoes = listar_execucoes(PASTA_RESULTADOS)
    abas = ["Visão geral", "Orçamento × qualidade", "Otimalidade e latência", "Perguntas", "Exemplo de Brasília"]
    if not execucoes:
        st.warning(f"Nenhum experimento em {PASTA_RESULTADOS}. Rode `uv run task avaliar` para gerar resultados.")
        aba_brasilia(tema)
        return

    metadados = {pasta: _metadados(pasta) for pasta in execucoes}
    # Abre na execução mais recente que tem respostas do LLM (as de --sem-llm só têm seleção).
    padrao = next((i for i, pasta in enumerate(execucoes) if metadados[pasta].get("modelo_gerador")), 0)
    with st.sidebar:
        pasta = st.selectbox(
            "Experimento", execucoes, index=padrao, format_func=lambda p: _rotulo_execucao(p, metadados[p])
        )
    linhas_todas, perguntas, desempenho, execucao = carregar_execucao(
        str(pasta), (pasta / "linhas.jsonl").stat().st_mtime
    )
    orcamentos = sorted({linha["orcamento"] for linha in linhas_todas})
    presentes = [a for a in ALGORITMOS if any(linha["algoritmo"] == a for linha in linhas_todas)]
    tipos = sorted({linha["tipo"] for linha in linhas_todas})

    with st.sidebar:
        orcamento = st.select_slider(
            "Orçamento T (tokens)", orcamentos, value=512 if 512 in orcamentos else orcamentos[len(orcamentos) // 2]
        )
        algoritmos = st.multiselect("Algoritmos", presentes, default=presentes, format_func=NOMES.get)
        tipos_escolhidos = st.multiselect("Tipo de pergunta", tipos, default=tipos)
        st.divider()
        st.caption(
            f"{execucao['perguntas']} perguntas de {execucao['fonte']} · k = {execucao['k']}  \n"
            f"Gerador: {execucao.get('modelo_gerador') or '—'} ({execucao.get('esforco_gerador') or '—'})  \n"
            f"Juiz: {execucao.get('modelo_juiz') or '—'}  \n"
            f"Embeddings: {execucao['embedder']} · tokens: {execucao['tokenizador']}"
        )
        gasto = execucao.get("gasto_api", {})
        if gasto:
            total = sum(g["custo_estimado_usd"] for g in gasto.values())
            chamadas = sum(g["chamadas_api"] for g in gasto.values())
            st.caption(f"Chamadas novas à API nesta execução: {chamadas}, ~US$ {num(total, 4)} (o resto veio do cache)")
        for algoritmo in presentes:
            st.caption(f"**{NOMES[algoritmo]}**: {DESCRICOES[algoritmo]}")

    if not algoritmos:
        st.info("Escolha ao menos um algoritmo na barra lateral.")
        return
    linhas = [linha for linha in linhas_todas if linha["tipo"] in tipos_escolhidos]
    resumos = tabela_resumo(linhas, algoritmos, orcamentos)
    if resumos.empty:
        st.info("Nenhum resultado com os filtros atuais.")
        return

    visao, por_orcamento, otimalidade, aba_q, brasilia = st.tabs(abas)
    with visao:
        aba_visao_geral(resumos, algoritmos, orcamento, tema, bool(execucao.get("modelo_gerador")))
    with por_orcamento:
        aba_orcamento(resumos, algoritmos, tema)
    with otimalidade:
        aba_otimalidade(linhas, resumos, desempenho, algoritmos, orcamento, tema)
    with aba_q:
        aba_perguntas(linhas, perguntas, algoritmos, orcamento)
    with brasilia:
        aba_brasilia(tema)


main()
