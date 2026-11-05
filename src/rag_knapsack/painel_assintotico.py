"""Painel da análise assintótica: como tempo, operações, memória e qualidade crescem com n e T.

uv run task assintotica           # gera os dados (resultados/assintotica-<data>/)
uv run task painel-assintotico    # abre este painel
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from rag_knapsack.assintotica import (
    REGIME_N,
    REGIME_T,
    ajuste_exponencial,
    ajuste_n_e_t,
    ajuste_potencia,
    expoente_efetivo_nlogn,
)
from rag_knapsack.config import RAIZ

PASTA_RESULTADOS = Path(os.getenv("PAINEL_RESULTADOS", str(RAIZ / "resultados")))

ORDEM = ["topk", "greedy", "knapsack", "topk_corte", "forca_bruta"]
PRINCIPAIS = ORDEM[:4]
NOMES = {
    "topk": "Top-K",
    "greedy": "Greedy",
    "knapsack": "Knapsack DP",
    "topk_corte": "Top-K com corte",
    "forca_bruta": "Força bruta",
}
# Paleta categórica validada para daltonismo nos dois modos; a cor acompanha o algoritmo.
CORES = {
    "light": {
        "topk": "#2a78d6",
        "greedy": "#eb6834",
        "knapsack": "#1baf7a",
        "topk_corte": "#eda100",
        "forca_bruta": "#e87ba4",
    },
    "dark": {
        "topk": "#3987e5",
        "greedy": "#d95926",
        "knapsack": "#199e70",
        "topk_corte": "#c98500",
        "forca_bruta": "#d55181",
    },
}
# A forma é a codificação secundária (curvas que se sobrepõem, daltonismo, impressão).
FORMAS = {
    "topk": "circle",
    "greedy": "square",
    "knapsack": "diamond",
    "topk_corte": "triangle-up",
    "forca_bruta": "cross",
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
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Complexidade teórica do código em seletores.py e o expoente esperado em n (a) e em T (b).
TEORIA = {
    "topk": {"tempo": "Θ(n log n) pior, Θ(n) melhor", "memoria": "Θ(n)", "a": 1.0, "b": 0.0},
    "greedy": {"tempo": "Θ(n log n) pior, Θ(n) melhor", "memoria": "Θ(n)", "a": 1.0, "b": 0.0},
    "knapsack": {"tempo": "Θ(n·T) pior, Θ(n + T) melhor", "memoria": "Θ(n·T)", "a": 1.0, "b": 1.0},
    "topk_corte": {"tempo": "Θ(n log n) pior, Θ(n) melhor", "memoria": "Θ(n)", "a": 1.0, "b": 0.0},
    "forca_bruta": {"tempo": "Θ(2ⁿ)", "memoria": "Θ(2ⁿ)"},
}
MEDIDAS = {"tempo_us": "Tempo (µs)", "operacoes": "Operações", "memoria_kib": "Memória (KiB)"}


# ---------------------------------------------------------------------------------------
# Formatação e aparência

# Números nos gráficos no formato brasileiro (1.234,5).
LOCALE_PT_BR = {"number": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["R$ ", ""]}}


def potencias_de_10(minimo: float, maximo: float) -> list[float]:
    """Marcas do eixo log só nas potências de 10 (a grade densa do log atrapalha a leitura)."""
    inicio, fim = math.floor(math.log10(max(minimo, 1e-12))), math.ceil(math.log10(max(maximo, 1e-12)))
    return [10.0**k for k in range(inicio, fim + 1)]


def num(valor: float, casas: int = 2) -> str:
    """Número no formato brasileiro (1.234,56)."""
    return f"{valor:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")


_SOBRESCRITO = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def tempo_humano(ns: float) -> str:
    segundos = ns / 1e9
    if segundos < 1e-3:
        return f"{num(ns / 1e3, 1)} µs"
    if segundos < 1:
        return f"{num(segundos * 1e3, 1)} ms"
    if segundos < 120:
        return f"{num(segundos, 1)} s"
    if segundos < 7200:
        return f"{num(segundos / 60, 1)} min"
    if segundos < 172_800:
        return f"{num(segundos / 3600, 1)} h"
    anos = segundos / 31_557_600
    if anos < 2:
        return f"{num(segundos / 86_400, 0)} dias"
    if anos < 1e6:
        return f"{num(anos, 0)} anos"
    expoente = math.floor(math.log10(anos))
    return f"≈ {num(anos / 10**expoente, 1)}·10{str(expoente).translate(_SOBRESCRITO)} anos"


def tema_atual() -> str:
    return "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"


def cor(algoritmos: list[str], tema: str, legenda: bool = True) -> alt.Color:
    return alt.Color(
        "Algoritmo:N",
        scale=alt.Scale(domain=[NOMES[a] for a in algoritmos], range=[CORES[tema][a] for a in algoritmos]),
        legend=alt.Legend(title=None, orient="top", columns=5, labelLimit=160) if legenda else None,
    )


def forma(algoritmos: list[str], legenda: bool = True) -> alt.Shape:
    return alt.Shape(
        "Algoritmo:N",
        scale=alt.Scale(domain=[NOMES[a] for a in algoritmos], range=[FORMAS[a] for a in algoritmos]),
        legend=alt.Legend(title=None, orient="top", columns=5, labelLimit=160) if legenda else None,
    )


def configurar(grafico: alt.TopLevelMixin, tema: str) -> alt.TopLevelMixin:
    tinta = TINTA[tema]
    return (
        grafico.properties(padding={"left": 20, "right": 12, "top": 8, "bottom": 4})
        .configure(background="transparent", font="sans-serif", locale=LOCALE_PT_BR)
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
        .configure_legend(labelColor=tinta["secundario"], labelFontSize=12, symbolStrokeWidth=2, symbolOpacity=1)
        .configure_title(color=tinta["texto"], anchor="start", fontSize=14, fontWeight="normal")
        .configure_header(labelColor=tinta["secundario"], titleColor=tinta["secundario"], labelFontSize=13)
    )


def mostrar(grafico: alt.TopLevelMixin) -> None:
    st.altair_chart(grafico, width="stretch", theme=None)


def eixo_x_algoritmo(algoritmos: list[str]) -> alt.X:
    """Nas comparações os algoritmos ficam no eixo X."""
    # Nomes compostos em duas linhas, para não se sobreporem em gráficos estreitos.
    quebras = (
        "datum.label == 'Top-K com corte' ? ['Top-K', 'com corte'] : "
        "datum.label == 'Knapsack DP' ? ['Knapsack', 'DP'] : datum.label"
    )
    return alt.X(
        "Algoritmo:N",
        sort=[NOMES[a] for a in algoritmos],
        title=None,
        axis=alt.Axis(labelAngle=0, labelExpr=quebras, labelLineHeight=13, labelLimit=120),
    )


# ---------------------------------------------------------------------------------------
# Dados


def listar_execucoes(pasta: Path) -> list[Path]:
    if not pasta.exists():
        return []
    return sorted((p for p in pasta.glob("assintotica-*") if (p / "medidas.jsonl").exists()), reverse=True)


def _ler_jsonl(caminho: Path) -> pd.DataFrame:
    if not caminho.exists() or caminho.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(caminho, lines=True)


def _agregar(df: pd.DataFrame, campo: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["varredura", "algoritmo", "n", "T", "mediana", "p25", "p75", "media", "casos"])
    grupos = df.groupby(["varredura", "algoritmo", "n", "T"])[campo]
    tabela = grupos.agg(
        mediana="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75), media="mean", casos="count"
    ).reset_index()
    tabela["Algoritmo"] = tabela["algoritmo"].map(NOMES)
    return tabela


@st.cache_data(show_spinner="Carregando as medidas...")
def carregar(pasta: str, marca: float) -> dict:
    caminho = Path(pasta)
    medidas = _ler_jsonl(caminho / "medidas.jsonl")
    medidas["tempo_us"] = medidas["tempo_ns"] / 1000
    medidas["Algoritmo"] = medidas["algoritmo"].map(NOMES)
    memoria = _ler_jsonl(caminho / "memoria.jsonl")
    if not memoria.empty:
        memoria["Algoritmo"] = memoria["algoritmo"].map(NOMES)
    return {
        "medidas": medidas,
        "memoria": memoria,
        "execucao": json.loads((caminho / "execucao.json").read_text(encoding="utf-8")),
        "tempo_us": _agregar(medidas, "tempo_us"),
        "operacoes": _agregar(medidas, "operacoes"),
        "memoria_kib": _agregar(memoria, "memoria_kib"),
        "cenarios": _agregar_por(medidas, "cenarios", ["algoritmo", "cenario", "n"]),
        "escala": _agregar_por(medidas, "escala", ["algoritmo", "fator"]),
    }


def _agregar_por(medidas: pd.DataFrame, varredura: str, chaves: list[str]) -> pd.DataFrame:
    """Medianas de tempo e operações de uma varredura extra (melhor/pior caso, escala dos pesos)."""
    if medidas.empty or "varredura" not in medidas or not (medidas["varredura"] == varredura).any():
        return pd.DataFrame()
    dados = medidas[medidas["varredura"] == varredura]
    tabela = dados.groupby(chaves)[["tempo_us", "operacoes", "celulas", "comparacoes", "T"]].median().reset_index()
    tabela["Algoritmo"] = tabela["algoritmo"].map(NOMES)
    return tabela


def principal(tabela: pd.DataFrame, algoritmos: list[str]) -> pd.DataFrame:
    return tabela[(tabela["varredura"] == "principal") & tabela["algoritmo"].isin(algoritmos)]


def ajustes_por_algoritmo(tabelas: dict, algoritmos: list[str], n_minimo: int = 0, t_minimo: int = 0) -> pd.DataFrame:
    """Expoentes a (em n) e b (em T) de y ≈ c·n^a·T^b, sobre as medianas da varredura principal.

    `n_minimo` e `t_minimo` restringem o ajuste às entradas grandes (o regime assintótico).
    """
    linhas = []
    for medida, rotulo in MEDIDAS.items():
        tabela = principal(tabelas[medida], algoritmos)
        tabela = tabela[(tabela["n"] >= n_minimo) & (tabela["T"] >= t_minimo)]
        for algoritmo in algoritmos:
            pontos = tabela[tabela["algoritmo"] == algoritmo]
            if len(pontos) < 4 or pontos["n"].nunique() < 2 or pontos["T"].nunique() < 2:
                continue
            ajuste = ajuste_n_e_t(pontos["n"], pontos["T"], pontos["mediana"])
            linhas.append(
                {
                    "algoritmo": algoritmo,
                    "Algoritmo": NOMES[algoritmo],
                    "medida": medida,
                    "Medida": rotulo,
                    **ajuste,
                    "a_teorico": expoente_teorico_n(algoritmo, medida, sorted(pontos["n"].unique())),
                    "b_teorico": TEORIA[algoritmo]["b"],
                }
            )
    return pd.DataFrame(linhas)


def expoente_teorico_n(algoritmo: str, medida: str, ns: list[float]) -> float:
    """Expoente em n que a teoria prevê NESTA faixa de n.

    n·log n ajustado como n^a dá a um pouco acima de 1 (≈ 1 + 1/ln n), e é com esse valor que o
    tempo e as operações de quem ordena devem ser comparados; a memória deles é Θ(n), expoente 1.
    """
    if algoritmo in ("topk", "greedy", "topk_corte") and medida != "memoria_kib":
        return expoente_efetivo_nlogn(ns)
    return TEORIA[algoritmo]["a"]


def inclinacoes_locais(tabela: pd.DataFrame, eixo: str) -> pd.DataFrame:
    """Expoente local Δlog y / Δlog x entre pontos vizinhos da grade (por algoritmo e pelo outro eixo)."""
    outro = "T" if eixo == "n" else "n"
    linhas = []
    for (algoritmo, fixo), grupo in tabela.groupby(["algoritmo", outro]):
        grupo = grupo.sort_values(eixo)
        x, y = grupo[eixo].to_numpy(float), grupo["mediana"].to_numpy(float)
        for i in range(len(x) - 1):
            linhas.append(
                {
                    "algoritmo": algoritmo,
                    "Algoritmo": NOMES[algoritmo],
                    outro: fixo,
                    eixo: math.sqrt(x[i] * x[i + 1]),
                    "de": x[i],
                    "ate": x[i + 1],
                    "expoente_local": math.log(y[i + 1] / y[i]) / math.log(x[i + 1] / x[i]),
                }
            )
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------------------
# Gráficos


def colunas_expoente(ajustes: pd.DataFrame, campo: str, algoritmos: list[str], tema: str, titulo: str):
    """Expoente empírico por algoritmo (eixo X), com ±2 erros-padrão e o valor teórico (traço)."""
    erro = "erro_padrao_n" if campo == "expoente_n" else "erro_padrao_T"
    teorico = "a_teorico" if campo == "expoente_n" else "b_teorico"
    dados = ajustes.assign(baixo=ajustes[campo] - 2 * ajustes[erro], alto=ajustes[campo] + 2 * ajustes[erro])
    tinta = TINTA[tema]
    base = alt.Chart().encode(x=eixo_x_algoritmo(algoritmos))
    barras = base.mark_bar(size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y(f"{campo}:Q", title="expoente"),
        color=cor(algoritmos, tema),
        tooltip=[
            alt.Tooltip("Algoritmo:N"),
            alt.Tooltip("Medida:N"),
            alt.Tooltip(f"{campo}:Q", title="expoente", format=".3f"),
            alt.Tooltip(f"{erro}:Q", title="erro-padrão", format=".3f"),
            alt.Tooltip(f"{teorico}:Q", title="teoria"),
            alt.Tooltip("r2:Q", title="R²", format=".3f"),
        ],
    )
    intervalo = base.mark_rule(color=tinta["secundario"], strokeWidth=1.5).encode(y="baixo:Q", y2="alto:Q")
    referencia = base.mark_tick(color=tinta["texto"], thickness=2, size=34).encode(y=f"{teorico}:Q")
    rotulos = base.mark_text(dy=-10, fontSize=11, color=tinta["secundario"]).encode(
        y="alto:Q", text=alt.Text(f"{campo}:Q", format=".2f")
    )
    grafico = alt.layer(barras, intervalo, referencia, rotulos, data=dados).properties(width=230, height=230)
    return configurar(
        grafico.facet(column=alt.Column("Medida:N", title=None, sort=list(MEDIDAS.values()))).properties(title=titulo),
        tema,
    )


def pontos_maior_caso(tabela: pd.DataFrame, algoritmos: list[str], tema: str, titulo: str, eixo_y: str, formato: str):
    """Mediana (ponto) e intervalo interquartil (traço) por algoritmo, no eixo X; escala log."""
    tinta = TINTA[tema]
    marcas = potencias_de_10(float(tabela["p25"].min()), float(tabela["p75"].max()) * 3)
    base = alt.Chart(tabela).encode(x=eixo_x_algoritmo(algoritmos))
    intervalo = base.mark_rule(strokeWidth=2, color=tinta["secundario"]).encode(y="p25:Q", y2="p75:Q")
    pontos = base.mark_point(filled=True, size=130, stroke=tinta["superficie"], strokeWidth=2).encode(
        y=alt.Y(
            "mediana:Q",
            scale=alt.Scale(type="log", domain=[marcas[0], marcas[-1]]),
            title=eixo_y,
            axis=alt.Axis(values=marcas, format="~s"),
        ),
        color=cor(algoritmos, tema, legenda=False),
        shape=forma(algoritmos, legenda=False),
        tooltip=[
            alt.Tooltip("Algoritmo:N"),
            alt.Tooltip("mediana:Q", format=formato),
            alt.Tooltip("p25:Q", format=formato),
            alt.Tooltip("p75:Q", format=formato),
            alt.Tooltip("casos:Q"),
        ],
    )
    rotulos = base.mark_text(dy=-16, fontSize=11, color=tinta["secundario"]).encode(
        y="mediana:Q", text=alt.Text("mediana:Q", format=formato)
    )
    return configurar(alt.layer(intervalo, pontos, rotulos).properties(title=titulo, height=260), tema)


def curvas(
    tabela: pd.DataFrame,
    eixo: str,
    algoritmos: list[str],
    tema: str,
    titulo_y: str,
    formato: str,
    ajustes: pd.DataFrame | None = None,
    log_x: bool = True,
):
    """Mediana (linha e pontos) e faixa interquartil por algoritmo, em escala log-log."""
    tinta = TINTA[tema]
    rotulo_x = {"n": "n (candidatos)", "T": "T (orçamento em tokens)"}[eixo]
    valores = sorted(tabela[eixo].unique().tolist())
    base = alt.Chart(tabela).encode(
        x=alt.X(
            f"{eixo}:Q",
            title=rotulo_x,
            scale=alt.Scale(type="log" if log_x else "linear"),
            axis=alt.Axis(values=valores, format="d"),
        ),
        color=cor(algoritmos, tema),
    )
    faixa = base.mark_area(opacity=0.14).encode(y="p25:Q", y2="p75:Q")
    marcas = potencias_de_10(float(tabela["p25"].min()), float(tabela["p75"].max()))
    linha = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("mediana:Q", title=titulo_y, scale=alt.Scale(type="log"), axis=alt.Axis(values=marcas, format="~s"))
    )
    pontos = base.mark_point(filled=True, size=75, stroke=tinta["superficie"], strokeWidth=2).encode(
        y="mediana:Q",
        shape=forma(algoritmos),
        tooltip=[
            alt.Tooltip("Algoritmo:N"),
            alt.Tooltip(f"{eixo}:Q"),
            alt.Tooltip("mediana:Q", format=formato),
            alt.Tooltip("p25:Q", format=formato),
            alt.Tooltip("p75:Q", format=formato),
            alt.Tooltip("casos:Q"),
        ],
    )
    camadas = [faixa, linha, pontos]
    if ajustes is not None and not ajustes.empty:
        camadas.append(
            alt.Chart(ajustes)
            .mark_line(strokeWidth=1.5, strokeDash=[5, 4], opacity=0.9)
            .encode(
                x=f"{eixo}:Q",
                y="previsto:Q",
                color=cor(algoritmos, tema),
                detail="Algoritmo:N",
            )
        )
    return configurar(alt.layer(*camadas).properties(height=380), tema)


def linhas_simples(
    dados: pd.DataFrame,
    x: str,
    y: str,
    algoritmos: list[str],
    tema: str,
    titulo_x: str,
    titulo_y: str,
    formato: str,
    log_x: bool = True,
    log_y: bool = False,
    referencias=(),
):
    tinta = TINTA[tema]
    escala_y = alt.Scale(type="log") if log_y else alt.Scale(zero=False)
    eixo_y = (
        alt.Axis(values=potencias_de_10(float(dados[y].min()), float(dados[y].max())), format="~s")
        if log_y
        else alt.Axis()
    )
    base = alt.Chart(dados).encode(
        x=alt.X(
            f"{x}:Q",
            title=titulo_x,
            scale=alt.Scale(type="log" if log_x else "linear"),
            axis=alt.Axis(values=sorted(dados[x].unique().tolist()), format="d") if log_x else alt.Axis(),
        ),
        y=alt.Y(f"{y}:Q", title=titulo_y, scale=escala_y, axis=eixo_y),
        color=cor(algoritmos, tema),
    )
    camadas = [
        base.mark_line(strokeWidth=2),
        base.mark_point(filled=True, size=70, stroke=tinta["superficie"], strokeWidth=2).encode(
            shape=forma(algoritmos),
            tooltip=[alt.Tooltip("Algoritmo:N"), alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q", format=formato)],
        ),
    ]
    for valor, texto in referencias:
        regra = pd.DataFrame({"y": [valor], "texto": [texto]})
        camadas.append(alt.Chart(regra).mark_rule(color=tinta["mudo"], strokeWidth=1.2).encode(y="y:Q"))
        camadas.append(
            alt.Chart(regra)
            .mark_text(align="left", dx=4, dy=-7, fontSize=11, color=tinta["mudo"])
            .encode(y="y:Q", x=alt.value(0), text="texto:N")
        )
    return configurar(alt.layer(*camadas).properties(height=320), tema)


def mapa_de_calor(tabela: pd.DataFrame, tema: str, titulo: str, legenda: str, formato_texto) -> alt.Chart:
    tinta = TINTA[tema]
    dados = tabela.assign(texto=tabela["mediana"].map(formato_texto))
    minimo, maximo = float(dados["mediana"].min()), float(dados["mediana"].max())
    limiar = math.sqrt(minimo * maximo) if minimo > 0 else (minimo + maximo) / 2
    base = alt.Chart(dados).encode(
        x=alt.X("n:O", title="n (candidatos)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("T:O", title="T (tokens)", sort="descending"),
    )
    celulas = base.mark_rect(stroke=tinta["superficie"], strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color(
            "mediana:Q", title=legenda, scale=alt.Scale(type="log" if minimo > 0 else "linear", range=RAMPA_AZUL)
        ),
        tooltip=[alt.Tooltip("n:O"), alt.Tooltip("T:O"), alt.Tooltip("texto:N", title=legenda), alt.Tooltip("casos:Q")],
    )
    texto = base.mark_text(fontSize=10.5).encode(
        text="texto:N",
        color=alt.condition(alt.datum.mediana > limiar, alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return configurar(alt.layer(celulas, texto).properties(title=titulo, height=340), tema)


# ---------------------------------------------------------------------------------------
# Abas


def aba_resumo(tabelas: dict, algoritmos: list[str], execucao: dict, tema: str) -> None:
    st.markdown(
        f"**{execucao['casos']} casos** (perguntas do HotpotQA) × **n** ∈ {execucao['n']} × **T** ∈ {execucao['T']}, "
        "cada algoritmo medido em todas as combinações. Os expoentes vêm do ajuste **y ≈ c · nᵃ · Tᵇ** sobre as "
        "medianas dos casos. O traço preto é o que a teoria prevê **nesta faixa de n**: para n·log n, ajustado como "
        "nᵃ, isso dá a um pouco acima de 1 (≈ 1 + 1/ln n). A barra fina é ±2 erros-padrão."
    )
    regime = execucao.get("regime", {})
    valores_n, valores_t = sorted(execucao["n"]), sorted(execucao["T"])
    opcoes_n, opcoes_t = valores_n[:-1] or valores_n, valores_t[:-1] or valores_t  # sobram ao menos 2 pontos

    def padrao(opcoes: list[int], preferido: int) -> int:
        return preferido if preferido in opcoes else opcoes[len(opcoes) // 2]

    controle1, controle2 = st.columns(2)
    n_minimo = controle1.select_slider(
        "Ajustar a partir de n =",
        opcoes_n,
        value=padrao(opcoes_n, regime.get("n_minimo", REGIME_N)),
        key="n_minimo",
        help="A análise assintótica fala de entradas grandes: em n e T pequenos dominam custos fixos.",
    )
    t_minimo = controle2.select_slider(
        "Ajustar a partir de T =",
        opcoes_t,
        value=padrao(opcoes_t, regime.get("t_minimo", REGIME_T)),
        key="t_minimo",
        help="Com T pequeno, parte dos parágrafos não cabe e fica fora da DP (ver aba Teoria).",
    )
    ajustes = ajustes_por_algoritmo(tabelas, algoritmos, n_minimo, t_minimo)
    if ajustes.empty:
        st.info(
            "A grade desta execução é pequena demais para ajustar expoentes "
            "(são precisos ao menos 2 valores de n e de T)."
        )
    else:
        mostrar(
            colunas_expoente(ajustes, "expoente_n", algoritmos, tema, "Expoente em n (a): quanto cresce ao dobrar n")
        )
        mostrar(
            colunas_expoente(ajustes, "expoente_T", algoritmos, tema, "Expoente em T (b): quanto cresce ao dobrar T")
        )
        st.caption(
            "Ao dobrar a entrada, o custo é multiplicado por 2ᵃ (ou 2ᵇ): expoente 1 é linear, 0 é constante. Todo "
            "seletor precisa ler os n candidatos, então nenhum custo pode crescer menos que n no limite (Ω(n)): um "
            "expoente de tempo abaixo de 1 indica que o custo por operação ainda está mudando nessa faixa (custos "
            "fixos do Python ou, no Knapsack, a fração de melhorias, que cai com n)."
        )
        st.info(
            "**Operações × tempo.** As operações vêm de réplicas instrumentadas que executam os mesmos laços e "
            "ordenações do código (e escolhem exatamente os mesmos chunks, conferido em toda medição); o tempo é a "
            "validação independente. Na DP, o modelo conta toda célula como 1, mas uma célula que **melhora** a "
            "solução (grava valor e decisão) custa mais do que uma que não melhora. A fração de melhorias cai quando "
            "n cresce e sobe com T, o que puxa o expoente do tempo um pouco abaixo de 1 em n e acima de 1 em T. A "
            "aba Operações ajusta tempo ≈ c₁·células + c₂·melhorias e mostra que isso explica o desvio."
        )
    n_max, t_max = max(execucao["n"]), max(execucao["T"])
    st.markdown(f"#### No maior caso medido (n = {n_max}, T = {t_max})")
    colunas = st.columns(3, gap="large")
    for coluna, (medida, titulo, formato) in zip(
        colunas,
        [
            ("tempo_us", "Tempo de uma seleção (µs)", ",.1f"),
            ("operacoes", "Operações elementares", ",.0f"),
            ("memoria_kib", "Pico de memória (KiB)", ",.1f"),
        ],
        strict=True,
    ):
        tabela = principal(tabelas[medida], algoritmos)
        tabela = tabela[(tabela["n"] == n_max) & (tabela["T"] == t_max)]
        with coluna:
            if tabela.empty:
                st.caption(f"{titulo}: sem medidas neste ponto.")
            else:
                mostrar(pontos_maior_caso(tabela, algoritmos, tema, titulo, titulo, formato))
    st.markdown("#### Complexidade teórica e expoentes medidos")
    linhas = []
    for algoritmo in [*algoritmos, "forca_bruta"]:
        registro = {
            "Algoritmo": NOMES[algoritmo],
            "Tempo (teoria)": TEORIA[algoritmo]["tempo"],
            "Memória (teoria)": TEORIA[algoritmo]["memoria"],
        }
        for medida, rotulo in (("tempo_us", "tempo"), ("operacoes", "operações"), ("memoria_kib", "memória")):
            achado = (
                ajustes[(ajustes["algoritmo"] == algoritmo) & (ajustes["medida"] == medida)]
                if not ajustes.empty
                else ajustes
            )
            if len(achado):
                linha = achado.iloc[0]
                registro[f"{rotulo}: n^a · T^b"] = f"n^{num(linha['expoente_n'])} · T^{num(linha['expoente_T'])}"
                registro[f"R² ({rotulo})"] = round(float(linha["r2"]), 3)
        if algoritmo == "forca_bruta":
            for medida, rotulo in (("tempo_us", "tempo"), ("operacoes", "operações"), ("memoria_kib", "memória")):
                forca = tabelas[medida]
                forca = forca[(forca["varredura"] == "forca_bruta") & (forca["algoritmo"] == "forca_bruta")]
                if len(forca) >= 3:
                    exp = ajuste_exponencial(forca["n"], forca["mediana"])
                    registro[f"{rotulo}: n^a · T^b"] = f"{num(exp['base'])}ⁿ (exponencial)"
                    registro[f"R² ({rotulo})"] = round(exp["r2"], 3)
        linhas.append(registro)
    tabela_teoria = pd.DataFrame(linhas).astype(object)
    st.dataframe(tabela_teoria.where(tabela_teoria.notna(), "—"), hide_index=True)
    st.caption(
        f"Ajustes com n ≥ {n_minimo} e T ≥ {t_minimo}. Memória do Knapsack: n·(T+1) bytes da tabela de decisões "
        "mais ≈ 32·(T+1) bytes do vetor de melhores valores (floats do Python), que não depende de n; por isso o "
        "expoente em n só chega a 1 quando n ≫ 32."
    )


def aba_teoria() -> None:
    st.markdown(
        """
Notação: **n** é o número de chunks candidatos, **T** o orçamento em tokens, **k** o limite do Top-K, **m** quantos
chunks foram escolhidos (m ≤ n) e **n′** quantos têm wᵢ ≤ T. **O** é limite superior, **Ω** inferior e **Θ** os dois
juntos (limite justo). Todo custo abaixo é do código em `seletores.py`, contado em passos elementares.

| Algoritmo | Melhor caso | Pior caso | Memória extra | Quando acontece o melhor caso |
|---|---|---|---|---|
| Top-K | Θ(n) | Θ(n log n) | Θ(n) | candidatos já chegam ordenados por score (o Timsort faz n − 1 comparações) |
| Top-K com corte | Θ(n) | Θ(n log n) | Θ(n) | idem |
| Greedy | Θ(n) | Θ(n log n) | Θ(n) | candidatos já chegam ordenados por densidade e poucos são escolhidos |
| Knapsack DP | Θ(n + T) | Θ(n·T) | Θ(n·T) | nenhum chunk cabe em T (a DP não tem itens) |
| Força bruta | Θ(2ⁿ) | Θ(2ⁿ) | Θ(2ⁿ) | não há: sempre visita todos os subconjuntos |

Custo exato da DP para uma entrada qualquer: **Θ(n + T + Σ₍wᵢ≤T₎ (T − wᵢ + 1))**, entre os dois extremos da tabela.
As medidas da varredura principal representam o **caso típico** da distribuição do HotpotQA; a aba *Melhor × pior
caso* mede os extremos.
"""
    )
    st.markdown("##### Top-K, Top-K com corte e Greedy")
    st.markdown(
        "O Timsort faz Θ(n log n) comparações no pior caso e n − 1 quando a entrada já está ordenada. O resto é uma "
        "varredura Θ(n) e a reordenação final dos m escolhidos, Θ(m log m) ≤ Θ(n log n). O Top-K não olha T. O Greedy "
        "olha: com T maior, mais chunks são escolhidos e a reordenação final custa mais, mas como m ≤ n o total segue "
        "limitado por O(n log n), sem crescer indefinidamente com T."
    )
    st.latex(
        r"C_{\text{Top-K}}(n) = \underbrace{n\log_2 n}_{\text{ordenação}} + "
        r"\underbrace{O(n)}_{\text{varredura}} = \Theta(n \log n)"
    )
    st.markdown("##### Knapsack DP")
    st.markdown("Para cada item i e cada capacidade t, a recorrência decide se o item entra:")
    st.latex(
        r"\mathrm{melhor}_i[t] = \max\left(\mathrm{melhor}_{i-1}[t],\; \mathrm{melhor}_{i-1}[t - w_i] + v_i\right)"
    )
    st.markdown(
        "O laço interno percorre t de T até wᵢ, então o número de células visitadas é a soma abaixo. A réplica "
        "instrumentada conta as células executando o laço, e o resultado bate com a fórmula:"
    )
    st.latex(r"C_{\text{DP}}(n, T) = \sum_{i:\, w_i \le T} (T - w_i + 1) \;\le\; n\,(T+1) = O(n \cdot T)")
    st.markdown(
        "Quando T é pequeno perto dos pesos, muitos itens não cabem e saem da soma: o crescimento em T fica "
        "**acima de linear** até T ≫ wᵢ, quando o expoente local tende a 1. A tabela `pegou` guarda uma decisão "
        "por célula (para reconstruir a solução), então a memória também é Θ(n·T): são n·(T+1) bytes de decisões mais "
        "(T+1) floats do vetor de melhores valores (≈ 32 bytes cada no Python). O segundo termo não depende de n e "
        "equivale a uns 32 itens, por isso o expoente medido em n só se aproxima de 1 com n ≫ 32."
    )
    st.markdown("##### Por que Θ(n·T) não é polinomial: pseudo-polinomialidade")
    st.markdown(
        "O tamanho da entrada é o número de bits usados para escrevê-la. Um inteiro T ≥ 1 ocupa "
        "b = ⌊log₂ T⌋ + 1 bits (T = 512 ocupa 10 bits), logo T ≥ 2^(b−1) e:"
    )
    st.latex(r"n \cdot T \;\ge\; n \cdot 2^{\,b-1}")
    st.markdown(
        "O custo é polinomial no **valor** de T, mas exponencial no **tamanho** (em bits) de T. A aba "
        "*Pseudo-polinomial* mostra isso na prática: multiplicando todos os pesos e T por s, a solução não muda, "
        "mas o custo da DP cresce ∝ s (e o dos outros algoritmos fica igual)."
    )
    st.markdown("##### Complexidade do problema × complexidade do algoritmo")
    st.markdown(
        "- **Mochila 0/1** (versão de decisão: existe subconjunto com peso ≤ T e valor ≥ V?) é **NP-completa**. Ela é "
        "NP-completa no sentido **fraco**: admite algoritmo pseudo-polinomial (esta DP) e, para valores, um esquema "
        "de aproximação totalmente polinomial (FPTAS). Não se conhece algoritmo polinomial em n e log T, e encontrar "
        "um provaria P = NP.\n"
        "- **Limites inferiores** triviais: todo seletor precisa ler os n candidatos, Ω(n). Ordenar por comparações "
        "custa Ω(n log n) no pior caso, mas **o problema Top-K não exige ordenar tudo**: seleção em Θ(n) (quickselect "
        "ou mediana das medianas) mais a ordenação dos k escolhidos dá O(n + k log k). O Top-K implementado ordena "
        "tudo por simplicidade; o expoente medido reflete a implementação, não o limite do problema.\n"
        "- A DP tem limite inferior Ω(n′·T) **para este algoritmo** (ele visita toda célula); para o problema, não se "
        "conhece limite inferior que exclua algoritmos bem mais rápidos no caso geral."
    )
    st.markdown("##### Compromissos de espaço")
    st.markdown(
        "- **Só o valor ótimo**: a DP precisa apenas do vetor `melhor`, Θ(T) de memória. Guardar `pegou` (Θ(n·T)) é o "
        "preço de reconstruir **quais** chunks entram; com divisão e conquista (estilo Hirschberg) dá para reconstruir "
        "com Θ(n + T) de memória, ao custo de uma constante maior no tempo.\n"
        "- **Força bruta**: esta implementação guarda peso e valor de cada subconjunto (Θ(2ⁿ) de memória) para gastar "
        "O(1) por subconjunto. Enumerando em código de Gray, cada passo muda um item só e a memória cai para O(n), "
        "mantendo Θ(2ⁿ) de tempo."
    )
    st.markdown("##### Força bruta")
    st.markdown(
        "Há 2ⁿ subconjuntos. A implementação reaproveita o peso e o valor do subconjunto sem o bit mais baixo, "
        "então cada subconjunto custa O(1):"
    )
    st.latex(r"C_{\text{bruta}}(n) = 2^n - 1 = \Theta(2^n) \qquad (\text{ingênua: } \Theta(n \cdot 2^n))")
    st.markdown(
        "Teste de estabilização: a razão C(n + 2) ÷ C(n) deve tender a 2² = 4 quando os termos aditivos "
        "deixam de pesar."
    )
    st.markdown("##### Garantia do Greedy (qualidade, não tempo)")
    st.markdown(
        "Descarte primeiro os itens com wᵢ > T e os de score ≤ 0: nenhum deles está no ótimo, e o Greedy implementado "
        "nunca os escolhe. Entre os itens restantes, seja k o primeiro, na ordem de densidade, que não coube e "
        "v_max o maior score. A mochila fracionária (que pode levar parte de um item) é uma relaxação da 0/1, e o "
        "guloso por densidade a resolve de forma ótima: leva os mesmos itens que o Greedy levou antes de k e uma "
        "fração de k. Pulando e continuando, o Greedy só pode somar mais. Daí:"
    )
    st.latex(
        r"\mathrm{OPT} \;\le\; \mathrm{OPT}_{\text{frac}} \;\le\; "
        r"\mathrm{Greedy} + v_k \;\le\; \mathrm{Greedy} + v_{\max}"
    )
    st.markdown(
        "A perda do Greedy é no máximo o valor de um chunk que cabe. Quando T cresce em relação aos pesos, cabem "
        "muitos chunks e a soma ótima cresce, então a **razão** Greedy/OPT tende a 1: o guloso é assintoticamente "
        "ótimo nesse regime. O Top-K com k fixo **não** tem essa propriedade (ele não usa o orçamento que sobra). "
        "A aba *Qualidade × tamanho* confere as duas coisas nos dados."
    )
    st.markdown("##### O que esperar do experimento")
    st.markdown(
        "- **Expoente em n**: ≈ 1 + 1/ln n para quem ordena (n log n na faixa medida) e 1 para o Knapsack.\n"
        "- **Expoente em T**: ≈ 0 para Top-K e Top-K com corte; pequeno e positivo para o Greedy (reordenação dos "
        "escolhidos); ≈ 1 para o Knapsack (acima de 1 com T pequeno, pelo efeito dos pesos).\n"
        "- **Força bruta**: base 2 no ajuste c·baseⁿ quando n não é pequeno; razão C(n+2)/C(n) → 4.\n"
        "- Em n pequeno o tempo é dominado por custo fixo do Python (chamada de função, criação de objetos), "
        "por isso o expoente local do tempo é menor que o das operações no começo da curva."
    )


def _previsoes(tabela: pd.DataFrame, eixo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajuste de potência por algoritmo ao longo de um eixo; devolve curvas previstas e expoentes."""
    curvas_previstas, expoentes = [], []
    for algoritmo, grupo in tabela.groupby("algoritmo"):
        if grupo[eixo].nunique() < 2:
            continue
        ajuste = ajuste_potencia(grupo[eixo], grupo["mediana"])
        xs = np.geomspace(grupo[eixo].min(), grupo[eixo].max(), 30)
        curvas_previstas.append(
            pd.DataFrame(
                {eixo: xs, "previsto": ajuste["coeficiente"] * xs ** ajuste["expoente"], "Algoritmo": NOMES[algoritmo]}
            )
        )
        expoentes.append(
            {
                "Algoritmo": NOMES[algoritmo],
                "expoente": ajuste["expoente"],
                "erro-padrão": ajuste["erro_padrao"],
                "R²": ajuste["r2"],
            }
        )
    previstos = pd.concat(curvas_previstas) if curvas_previstas else pd.DataFrame()
    return previstos, pd.DataFrame(expoentes)


def _tabela_expoentes(expoentes: pd.DataFrame, eixo: str) -> None:
    if expoentes.empty:
        return
    st.dataframe(
        expoentes,
        hide_index=True,
        column_config={
            "expoente": st.column_config.NumberColumn(f"expoente em {eixo}", format="%.3f"),
            "erro-padrão": st.column_config.NumberColumn(format="%.3f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    if (expoentes["expoente"].abs() < 0.1).any():
        st.caption(
            "R² baixo com expoente perto de 0 é esperado: a curva é plana, então não há tendência para o ajuste "
            "explicar além do ruído de medição."
        )


def aba_crescimento(tabelas: dict, algoritmos: list[str], execucao: dict, tema: str, eixo: str) -> None:
    outro = "T" if eixo == "n" else "n"
    valores_outro = sorted(execucao[outro])
    padrao = 512 if outro == "T" and 512 in valores_outro else valores_outro[len(valores_outro) // 2]
    controle1, controle2 = st.columns([3, 2])
    fixo = controle1.select_slider(f"{outro} fixo", valores_outro, value=padrao, key=f"fixo_{eixo}")
    medida = controle2.segmented_control(
        "Medida",
        ["tempo_us", "operacoes"],
        format_func=MEDIDAS.get,
        default="tempo_us",
        required=True,
        key=f"medida_{eixo}",
    )
    tabela = principal(tabelas[medida], algoritmos)
    corte = tabela[tabela[outro] == fixo]
    if corte.empty:
        st.info("Sem medidas para esta combinação.")
        return
    previstos, expoentes = _previsoes(corte, eixo)
    formato = ",.1f" if medida == "tempo_us" else ",.0f"
    st.markdown(
        f"**{MEDIDAS[medida]} × {eixo}** com {outro} = {fixo}, em escala log-log: uma reta de inclinação a "
        f"significa crescimento {eixo}ᵃ. Linha cheia: mediana dos casos; faixa: 25% a 75% dos casos; "
        "tracejado: ajuste de potência."
    )
    mostrar(curvas(corte, eixo, algoritmos, tema, MEDIDAS[medida], formato, previstos))
    _tabela_expoentes(expoentes, eixo)
    st.markdown(f"##### Expoente local ao longo de {eixo}")
    st.caption(
        f"Inclinação entre pontos vizinhos da grade (Δlog y ÷ Δlog {eixo}). Se a teoria vale, o expoente local "
        "se estabiliza no valor assintótico quando a entrada cresce."
    )
    locais = inclinacoes_locais(corte, eixo)
    if not locais.empty:
        referencias = [(1.0, "linear (1)"), (0.0, "constante (0)")]
        mostrar(
            linhas_simples(
                locais,
                eixo,
                "expoente_local",
                algoritmos,
                tema,
                f"{eixo} (ponto médio do intervalo)",
                "expoente local",
                ".2f",
                referencias=referencias,
            )
        )
    if eixo == "T" and "knapsack" in algoritmos:
        st.info(
            "No Knapsack o expoente local em T começa acima de 1: com T pequeno, parte dos parágrafos não cabe "
            "e fica fora da DP. Conforme T cresce, todos entram e as células ficam ≈ n·T, com expoente → 1."
        )
    with st.expander("Todas as curvas: um gráfico por valor fixo"):
        mostrar(
            configurar(
                alt.Chart(tabela)
                .mark_line(strokeWidth=1.8, point=alt.OverlayMarkDef(size=30, filled=True))
                .encode(
                    x=alt.X(f"{eixo}:Q", scale=alt.Scale(type="log"), title=eixo),
                    y=alt.Y("mediana:Q", scale=alt.Scale(type="log"), title=MEDIDAS[medida]),
                    color=cor(algoritmos, tema),
                    tooltip=[
                        alt.Tooltip("Algoritmo:N"),
                        alt.Tooltip(f"{eixo}:Q"),
                        alt.Tooltip(f"{outro}:Q"),
                        alt.Tooltip("mediana:Q", format=formato),
                    ],
                )
                .properties(width=210, height=170)
                .facet(facet=alt.Facet(f"{outro}:O", title=f"{outro} fixo"), columns=4),
                tema,
            )
        )


def aba_operacoes(tabelas: dict, algoritmos: list[str], execucao: dict, tema: str, medidas: pd.DataFrame) -> None:
    st.markdown(
        "As **operações elementares** vêm de réplicas instrumentadas que executam os mesmos laços e ordenações de "
        "`seletores.py` (comparações feitas pelo Timsort, células da DP, passos de laço, posições de memória zeradas). "
        "Em toda medição o experimento confere que a réplica escolhe exatamente os mesmos chunks que o código real. "
        'A contagem não depende da máquina: é a medida mais próxima da análise assintótica "de livro"; o tempo de '
        "relógio é a validação independente."
    )
    valores_t = sorted(execucao["T"])
    fixo = st.select_slider("T fixo", valores_t, value=512 if 512 in valores_t else valores_t[0], key="t_operacoes")
    operacoes = principal(tabelas["operacoes"], algoritmos)
    tempos = principal(tabelas["tempo_us"], algoritmos)
    corte = operacoes[operacoes["T"] == fixo].copy()

    def modelo(linha) -> float:
        n = linha["n"]
        return n * linha["T"] if linha["algoritmo"] == "knapsack" else n * math.log2(max(n, 2))

    corte["modelo"] = corte.apply(modelo, axis=1)
    corte["razao"] = corte["mediana"] / corte["modelo"]
    esquerda, direita = st.columns(2, gap="large")
    with esquerda:
        st.markdown("**Operações ÷ modelo teórico**")
        st.caption(
            "Modelo: n·log₂n para os que ordenam; n·T para o Knapsack. Uma curva que se estabiliza confirma a ordem de "
            "grandeza; a altura é a constante oculta (no Knapsack, inclui zerar a tabela de decisões)."
        )
        mostrar(
            linhas_simples(
                corte,
                "n",
                "razao",
                algoritmos,
                tema,
                "n (candidatos)",
                "operações ÷ modelo",
                ".3f",
                referencias=[(1.0, "= modelo")],
            )
        )
    with direita:
        juntas = operacoes.merge(tempos, on=["varredura", "algoritmo", "Algoritmo", "n", "T"], suffixes=("_op", "_t"))
        juntas = juntas[juntas["T"] == fixo].assign(ns_por_operacao=lambda d: d["mediana_t"] * 1000 / d["mediana_op"])
        st.markdown("**Tempo por operação (ns)**")
        st.caption(
            "Se o tempo é proporcional às operações, esta razão fica constante: o custo fixo do Python pesa em n "
            "pequeno e se dilui quando n cresce."
        )
        mostrar(
            linhas_simples(
                juntas, "n", "ns_por_operacao", algoritmos, tema, "n (candidatos)", "ns por operação", ".1f", log_y=True
            )
        )
    if "knapsack" not in algoritmos:
        return
    dp = medidas[(medidas["varredura"] == "principal") & (medidas["algoritmo"] == "knapsack")]
    por_config = dp.groupby(["n", "T"])[["celulas", "melhorias", "tempo_ns"]].median().reset_index()
    por_config["casos"] = dp.groupby(["n", "T"]).size().to_numpy()
    st.markdown("##### Knapsack: células visitadas ÷ (n·T)")
    st.caption("Abaixo de 1 quando T é pequeno (itens que não cabem ficam fora da DP); tende a 1 quando T ≫ pesos.")
    mostrar(
        mapa_de_calor(
            por_config.assign(mediana=por_config["celulas"] / (por_config["n"] * por_config["T"])),
            tema,
            "células ÷ (n·T)",
            "razão",
            lambda v: num(v, 2),
        )
    )
    st.markdown("##### Knapsack: por que o tempo não segue exatamente as células")
    por_config = por_config[por_config["celulas"] > 0].copy()
    por_config["fracao_melhorias"] = por_config["melhorias"] / por_config["celulas"]
    esquerda, direita = st.columns(2, gap="large")
    with esquerda:
        st.caption(
            "Fração das células que **melhoram** a solução (e gravam valor e decisão). Ela cai quando n cresce: os "
            "itens processados por último raramente melhoram o ótimo."
        )
        mostrar(
            mapa_de_calor(
                por_config.assign(mediana=por_config["fracao_melhorias"]),
                tema,
                "melhorias ÷ células",
                "fração",
                lambda v: f"{v:.0%}",
            )
        )
    with direita:
        x_celulas = por_config["celulas"].to_numpy(float)
        x_melhorias = por_config["melhorias"].to_numpy(float)
        y = por_config["tempo_ns"].to_numpy(float)
        so_celulas = np.linalg.lstsq(np.column_stack([x_celulas, np.ones_like(y)]), y, rcond=None)[0]
        com_melhorias = np.linalg.lstsq(np.column_stack([x_celulas, x_melhorias, np.ones_like(y)]), y, rcond=None)[0]

        def erro_relativo(previsto: np.ndarray) -> float:
            return float(np.median(np.abs(previsto - y) / y))

        erro_1 = erro_relativo(np.column_stack([x_celulas, np.ones_like(y)]) @ so_celulas)
        erro_2 = erro_relativo(np.column_stack([x_celulas, x_melhorias, np.ones_like(y)]) @ com_melhorias)
        st.caption("Modelos de custo ajustados ao tempo mediano de cada (n, T), por mínimos quadrados:")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "modelo": "c₀ + c₁·células",
                        "erro relativo mediano": erro_1,
                        "ns por célula": so_celulas[0],
                        "ns por melhoria": float("nan"),
                    },
                    {
                        "modelo": "c₀ + c₁·células + c₂·melhorias",
                        "erro relativo mediano": erro_2,
                        "ns por célula": com_melhorias[0],
                        "ns por melhoria": com_melhorias[1],
                    },
                ]
            ),
            hide_index=True,
            column_config={
                "ns por célula": st.column_config.NumberColumn(format="%.1f"),
                "ns por melhoria": st.column_config.NumberColumn(format="%.1f"),
                "erro relativo mediano": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "Se o segundo modelo erra bem menos, o desvio do tempo em relação a Θ(n·T) vem do custo extra das "
            "melhorias, não de uma ordem de crescimento diferente: as células seguem Θ(n·T)."
        )


def aba_mapa(tabelas: dict, algoritmos: list[str], tema: str) -> None:
    escolhido = st.segmented_control(
        "Algoritmo",
        algoritmos,
        format_func=NOMES.get,
        default="knapsack" if "knapsack" in algoritmos else algoritmos[0],
        required=True,
        key="algoritmo_mapa",
    )
    tempo = principal(tabelas["tempo_us"], [escolhido])
    st.markdown(
        f"**{NOMES[escolhido]}: tempo mediano de uma seleção em cada (n, T).** Cor em escala logarítmica. "
        "Para o Knapsack a cor cresce nas duas direções (Θ(n·T)); para os outros, só ao longo de n."
    )
    mostrar(mapa_de_calor(tempo, tema, f"{NOMES[escolhido]}: tempo (µs)", "µs", lambda v: num(v, 1 if v < 100 else 0)))
    operacoes = principal(tabelas["operacoes"], [escolhido])
    mostrar(
        mapa_de_calor(
            operacoes,
            tema,
            f"{NOMES[escolhido]}: operações elementares",
            "operações",
            lambda v: f"{v / 1000:,.0f}k".replace(",", ".") if v >= 10_000 else num(v, 0),
        )
    )


def config_forca_bruta(execucao: dict) -> dict:
    grade = execucao.get("grade", {})
    antiga = execucao.get("forca_bruta", {})
    return {
        "n": grade.get("n_forca_bruta", antiga.get("n", [])),
        "T": grade.get("t_forca_bruta", antiga.get("T", 512)),
        "casos": grade.get("casos_forca_bruta", antiga.get("casos", 0)),
    }


def aba_forca_bruta(tabelas: dict, execucao: dict, tema: str) -> None:
    forca = tabelas["tempo_us"]
    forca = forca[forca["varredura"] == "forca_bruta"]
    if forca.empty:
        st.info("Esta execução não tem a varredura da força bruta.")
        return
    config = config_forca_bruta(execucao)
    st.markdown(
        f"Força bruta e Knapsack DP nas mesmas instâncias ({config['casos']} casos, T = {config['T']}, "
        f"n até {max(config['n'])}). Escala log no eixo Y: uma **reta** é crescimento exponencial. "
        "Os dois sempre chegam ao mesmo valor ótimo (o experimento confere)."
    )
    so_bruta = forca[forca["algoritmo"] == "forca_bruta"].sort_values("n")
    valores_n = sorted(so_bruta["n"].unique().tolist())
    opcoes = valores_n[:-2] or valores_n
    padrao = execucao.get("regime", {}).get("n_minimo_forca_bruta", 12)
    n_minimo = st.select_slider(
        "Ajustar a base a partir de n =",
        opcoes,
        value=padrao if padrao in opcoes else opcoes[len(opcoes) // 2],
        key="n_minimo_forca",
        help="Em n pequeno os termos aditivos ainda pesam e puxam a base para baixo de 2.",
    )
    ajustados = so_bruta[so_bruta["n"] >= n_minimo]
    exp = ajuste_exponencial(ajustados["n"], ajustados["mediana"])
    dp = forca[forca["algoritmo"] == "knapsack"]
    potencia_dp = ajuste_potencia(dp["n"], dp["mediana"])
    algoritmos = ["knapsack", "forca_bruta"]
    xs = np.arange(min(config["n"]), 31)
    projecao = pd.DataFrame(
        {"n": xs, "previsto": exp["coeficiente"] * exp["base"] ** xs, "Algoritmo": NOMES["forca_bruta"]}
    )
    tinta = TINTA[tema]
    base = alt.Chart(forca).encode(
        x=alt.X("n:Q", title="n (candidatos)", scale=alt.Scale(domain=[min(xs), max(xs)])),
        color=cor(algoritmos, tema),
    )
    grafico = alt.layer(
        base.mark_area(opacity=0.14).encode(y="p25:Q", y2="p75:Q"),
        base.mark_line(strokeWidth=2).encode(
            y=alt.Y(
                "mediana:Q",
                scale=alt.Scale(type="log"),
                title="tempo (µs)",
                axis=alt.Axis(
                    values=potencias_de_10(float(forca["p25"].min()), float(projecao["previsto"].max())), format="~s"
                ),
            )
        ),
        base.mark_point(filled=True, size=75, stroke=tinta["superficie"], strokeWidth=2).encode(
            y="mediana:Q",
            shape=forma(algoritmos),
            tooltip=[alt.Tooltip("Algoritmo:N"), alt.Tooltip("n:Q"), alt.Tooltip("mediana:Q", format=",.1f")],
        ),
        alt.Chart(projecao)
        .mark_line(strokeDash=[5, 4], strokeWidth=1.5)
        .encode(x="n:Q", y="previsto:Q", color=cor(algoritmos, tema)),
    ).properties(height=380)
    mostrar(configurar(grafico, tema))
    st.caption(
        f"Tracejado: projeção da força bruta, tempo ≈ c·{num(exp['base'], 3)}ⁿ ajustado com n ≥ {n_minimo} "
        f"(R² = {num(exp['r2'], 3)})."
    )
    metricas = st.columns(3)
    metricas[0].metric(
        "Base do crescimento da força bruta", num(exp["base"], 3), help="Teoria: 2 (cada item dobra os subconjuntos)"
    )
    metricas[1].metric("Expoente da DP em n (T fixo)", num(potencia_dp["expoente"]), help="Teoria: 1")
    metricas[2].metric(
        "Cada 2 itens a mais multiplicam a força bruta por", f"{num(exp['base'] ** 2, 2)}×", help="Teoria: 2² = 4"
    )
    st.markdown("##### Teste de estabilização: razão C(n + 2) ÷ C(n)")
    st.caption(
        "Razão entre os custos medianos de n + 2 e n. Para Θ(2ⁿ) ela deve se aproximar de 4 quando n cresce; "
        "valores menores no começo são o peso dos termos aditivos."
    )
    razoes = []
    for medida, rotulo in (("tempo_us", "tempo"), ("operacoes", "operações")):
        tabela = tabelas[medida]
        tabela = tabela[(tabela["varredura"] == "forca_bruta") & (tabela["algoritmo"] == "forca_bruta")].sort_values(
            "n"
        )
        ns, ys = tabela["n"].to_numpy(), tabela["mediana"].to_numpy(float)
        for i in range(len(ns) - 1):
            if ns[i + 1] - ns[i] == 2:
                razoes.append({"n": int(ns[i + 1]), "razao": ys[i + 1] / ys[i], "Medida": rotulo})
    if razoes:
        dados = pd.DataFrame(razoes)
        linhas = (
            alt.Chart(dados)
            .mark_line(strokeWidth=2, color=CORES[tema]["forca_bruta"], point=True)
            .encode(
                x=alt.X("n:Q", title="n (de n − 2 para n)"),
                y=alt.Y("razao:Q", title="C(n) ÷ C(n − 2)", scale=alt.Scale(zero=False)),
                strokeDash=alt.StrokeDash("Medida:N", legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("Medida:N"), alt.Tooltip("n:Q"), alt.Tooltip("razao:Q", format=".3f")],
            )
        )
        quatro = alt.Chart(pd.DataFrame({"y": [4.0]})).mark_rule(color=tinta["mudo"]).encode(y="y:Q")
        mostrar(configurar(alt.layer(quatro, linhas).properties(height=260), tema))
    st.markdown("##### Extrapolação: quanto levaria com mais candidatos")
    linhas_tabela = []
    for n in (20, 30, 40, 50, 100, 200, 800):
        bruta_ns = exp["coeficiente"] * exp["base"] ** n * 1000
        dp_ns = potencia_dp["coeficiente"] * n ** potencia_dp["expoente"] * 1000
        linhas_tabela.append(
            {"n": n, "Força bruta (projeção)": tempo_humano(bruta_ns), "Knapsack DP (projeção)": tempo_humano(dp_ns)}
        )
    st.dataframe(pd.DataFrame(linhas_tabela), hide_index=True)
    st.caption(
        f"Projeções a partir dos ajustes acima (T = {config['T']}), nesta máquina. Com n = 200 a força bruta passaria "
        "muito da idade do universo (~1,4·10¹⁰ anos); a DP mede-se em milissegundos."
    )


def aba_memoria(tabelas: dict, algoritmos: list[str], execucao: dict, tema: str) -> None:
    memoria = principal(tabelas["memoria_kib"], algoritmos)
    if memoria.empty:
        st.info("Esta execução não mediu memória.")
        return
    st.markdown(
        f"Pico de memória alocada durante uma seleção (tracemalloc), em {_grade(execucao, 'casos_memoria', '?')} "
        "casos por "
        "configuração. Teoria: Θ(n) para os que ordenam; Θ(n·T) para o Knapsack (a tabela de decisões)."
    )
    st.caption(
        "Nesta faixa de n, a memória por elemento dos algoritmos que ordenam sobe em degraus por detalhes do CPython: "
        "inteiros até 256 são reaproveitados (acima disso cada posição vira um objeto novo) e as tabelas de "
        "dicionário dobram de tamanho e alargam os índices de 1 para 2 bytes. Por isso o expoente medido fica um "
        "pouco acima de 1; como os degraus são finitos, a ordem continua Θ(n). No Knapsack, o termo (T+1) do vetor "
        "de melhores valores não depende de n e puxa o expoente em n para baixo de 1."
    )
    esquerda, direita = st.columns(2, gap="large")
    valores_t, valores_n = sorted(execucao["T"]), sorted(execucao["n"])
    with esquerda:
        t_fixo = st.select_slider("T fixo", valores_t, value=valores_t[-1], key="t_memoria")
        corte = memoria[memoria["T"] == t_fixo]
        previstos, expoentes = _previsoes(corte, "n")
        mostrar(curvas(corte, "n", algoritmos, tema, "memória (KiB)", ",.1f", previstos))
        _tabela_expoentes(expoentes, "n")
    with direita:
        n_fixo = st.select_slider("n fixo", valores_n, value=valores_n[-1], key="n_memoria")
        corte = memoria[memoria["n"] == n_fixo]
        previstos, expoentes = _previsoes(corte, "T")
        mostrar(curvas(corte, "T", algoritmos, tema, "memória (KiB)", ",.1f", previstos))
        _tabela_expoentes(expoentes, "T")
    forca = tabelas["memoria_kib"]
    forca = forca[(forca["varredura"] == "forca_bruta") & (forca["algoritmo"] == "forca_bruta")]
    if not forca.empty:
        exp = ajuste_exponencial(forca["n"], forca["mediana"])
        st.caption(
            f"Força bruta: a memória cresce ≈ {num(exp['base'])}ⁿ (guarda peso e valor de cada subconjunto); "
            f"com n = {int(forca['n'].max())} já são {num(float(forca['mediana'].max()) / 1024, 1)} MiB."
        )


ROTULO_CENARIO = {"melhor": "melhor caso", "tipico": "típico (HotpotQA)", "pior": "pior caso"}
DESCRICAO_CENARIO = {
    "topk": "melhor: candidatos já ordenados por score · pior: ordem embaralhada",
    "topk_corte": "melhor: candidatos já ordenados por score · pior: ordem embaralhada",
    "greedy": "melhor: candidatos já ordenados por densidade · pior: ordem embaralhada",
    "knapsack": "melhor: nenhum chunk cabe em T · pior: todos cabem com folga (pesos ÷ 16)",
}


def _grade(execucao: dict, chave: str, padrao=None):
    return execucao.get("grade", {}).get(chave, execucao.get(chave, padrao))


def aba_cenarios(tabelas: dict, algoritmos: list[str], execucao: dict, tema: str) -> None:
    dados = tabelas["cenarios"]
    if dados.empty:
        st.info("Esta execução não tem a varredura de melhor e pior caso (rode de novo `uv run task assintotica`).")
        return
    dados = dados[dados["algoritmo"].isin(algoritmos)].copy()
    dados["Cenário"] = dados["cenario"].map(ROTULO_CENARIO)
    st.markdown(
        f"Cada algoritmo medido no seu **melhor caso**, no **caso típico** (a distribuição do HotpotQA) e no **pior "
        f"caso**, com T = {_grade(execucao, 't_cenarios', '?')} e {_grade(execucao, 'casos_cenarios', '?')} casos. "
        "A teoria prevê:"
    )
    st.markdown(
        "\n".join(f"- **{NOMES[a]}**: {DESCRICAO_CENARIO[a]}" for a in algoritmos if a in DESCRICAO_CENARIO)
        + "\n\nPara quem ordena, o melhor caso é Θ(n) (o Timsort faz n − 1 comparações numa entrada já ordenada) e o "
        "pior Θ(n log n). Para o Knapsack, o melhor caso é Θ(n + T) (sem itens na DP) e o pior Θ(n·T)."
    )
    medida = st.segmented_control(
        "Medida",
        ["operacoes", "tempo_us"],
        format_func=MEDIDAS.get,
        default="operacoes",
        required=True,
        key="medida_cenarios",
    )
    ordem_cenarios = [ROTULO_CENARIO[c] for c in ("melhor", "tipico", "pior")]
    tinta = TINTA[tema]
    grafico = (
        alt.Chart(dados)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(filled=True, size=60, stroke=tinta["superficie"]))
        .encode(
            x=alt.X(
                "n:Q",
                scale=alt.Scale(type="log"),
                title="n (candidatos)",
                axis=alt.Axis(values=sorted(dados["n"].unique().tolist()), format="d"),
            ),
            y=alt.Y(
                f"{medida}:Q",
                scale=alt.Scale(type="log"),
                title=MEDIDAS[medida],
                axis=alt.Axis(
                    values=potencias_de_10(float(dados[medida].min()), float(dados[medida].max())), format="~s"
                ),
            ),
            color=cor(algoritmos, tema, legenda=False),
            strokeDash=alt.StrokeDash(
                "Cenário:N",
                sort=ordem_cenarios,
                legend=alt.Legend(
                    title=None,
                    orient="top",
                    symbolType="stroke",
                    symbolStrokeColor=tinta["secundario"],
                    symbolStrokeWidth=2,
                ),
                scale=alt.Scale(domain=ordem_cenarios, range=[[2, 3], [1, 0], [7, 3]]),
            ),
            tooltip=[
                alt.Tooltip("Algoritmo:N"),
                alt.Tooltip("Cenário:N"),
                alt.Tooltip("n:Q"),
                alt.Tooltip(f"{medida}:Q", format=",.1f"),
            ],
        )
        .properties(width=380, height=210)
        .facet(facet=alt.Facet("Algoritmo:N", sort=[NOMES[a] for a in algoritmos], title=None), columns=2)
        .resolve_scale(y="independent")
    )
    mostrar(configurar(grafico, tema))
    st.caption("Cada painel tem a própria escala Y (log): o que importa é a inclinação de cada cenário.")
    linhas = []
    for (algoritmo, cenario), grupo in dados.groupby(["algoritmo", "cenario"]):
        if grupo["n"].nunique() < 2:
            continue
        ajuste = ajuste_potencia(grupo["n"], grupo[medida])
        linhas.append(
            {
                "algoritmo": algoritmo,
                "Algoritmo": NOMES[algoritmo],
                "Cenário": ROTULO_CENARIO[cenario],
                "expoente": ajuste["expoente"],
                "R²": ajuste["r2"],
            }
        )
    if not linhas:
        return
    expoentes = pd.DataFrame(linhas)
    st.markdown(f"**Expoente em n por cenário ({MEDIDAS[medida].lower()})**")
    base = alt.Chart(expoentes).encode(
        x=eixo_x_algoritmo(algoritmos),
        xOffset=alt.XOffset("Cenário:N", sort=ordem_cenarios, scale=alt.Scale(paddingInner=0.15)),
    )
    barras = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("expoente:Q", title="expoente em n"),
        color=cor(algoritmos, tema, legenda=False),
        opacity=alt.Opacity(
            "Cenário:N",
            sort=ordem_cenarios,
            scale=alt.Scale(domain=ordem_cenarios, range=[0.4, 0.7, 1.0]),
            legend=alt.Legend(title=None, orient="top", symbolFillColor=tinta["secundario"]),
        ),
        tooltip=[
            alt.Tooltip("Algoritmo:N"),
            alt.Tooltip("Cenário:N"),
            alt.Tooltip("expoente:Q", format=".3f"),
            alt.Tooltip("R²:Q", format=".3f"),
        ],
    )
    rotulos = base.mark_text(dy=-8, fontSize=10.5, color=tinta["secundario"]).encode(
        y="expoente:Q", text=alt.Text("expoente:Q", format=".2f")
    )
    um = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=tinta["mudo"]).encode(y="y:Q")
    mostrar(configurar(alt.layer(um, barras, rotulos).properties(height=280), tema))
    st.caption(
        "Barras mais transparentes = melhor caso; opacas = pior caso. A linha marca expoente 1 (linear). No Knapsack, "
        "o melhor caso cresce menos que n porque o custo Θ(n + T) com T fixo é dominado por T em n pequeno."
    )


def aba_escala(tabelas: dict, algoritmos: list[str], execucao: dict, tema: str) -> None:
    dados = tabelas["escala"]
    if dados.empty:
        st.info("Esta execução não tem a varredura de escala dos pesos (rode de novo `uv run task assintotica`).")
        return
    dados = dados[dados["algoritmo"].isin(algoritmos)].copy()
    n_escala, t_escala = _grade(execucao, "n_escala", "?"), _grade(execucao, "t_escala", 512)
    st.markdown(
        f"**Pseudo-polinomialidade na prática.** Mesma instância (n = {n_escala}), com todos os pesos e o orçamento "
        f"multiplicados por s (T = {t_escala}·s). O problema é o mesmo e a solução ótima não muda (o experimento "
        "confere), mas o número de bits de T cresce: cada vez que s dobra, T ganha 1 bit. A DP visita Θ(n·T) células, "
        "então dobra de custo a cada bit; os outros algoritmos não dependem de T."
    )
    dados["bits_T"] = dados["T"].map(lambda t: int(math.floor(math.log2(t))) + 1)
    medida = st.segmented_control(
        "Medida",
        ["operacoes", "tempo_us"],
        format_func=MEDIDAS.get,
        default="operacoes",
        required=True,
        key="medida_escala",
    )
    tinta = TINTA[tema]
    base = alt.Chart(dados).encode(
        x=alt.X(
            "fator:Q",
            scale=alt.Scale(type="log", base=2),
            title="fator s (pesos e T × s)",
            axis=alt.Axis(values=sorted(dados["fator"].unique().tolist()), format="d"),
        ),
        color=cor(algoritmos, tema),
    )
    grafico = alt.layer(
        base.mark_line(strokeWidth=2).encode(
            y=alt.Y(
                f"{medida}:Q",
                scale=alt.Scale(type="log"),
                title=MEDIDAS[medida],
                axis=alt.Axis(
                    values=potencias_de_10(float(dados[medida].min()), float(dados[medida].max())), format="~s"
                ),
            )
        ),
        base.mark_point(filled=True, size=75, stroke=tinta["superficie"], strokeWidth=2).encode(
            y=f"{medida}:Q",
            shape=forma(algoritmos),
            tooltip=[
                alt.Tooltip("Algoritmo:N"),
                alt.Tooltip("fator:Q", title="s"),
                alt.Tooltip("T:Q"),
                alt.Tooltip("bits_T:Q", title="bits de T"),
                alt.Tooltip(f"{medida}:Q", format=",.1f"),
            ],
        ),
    ).properties(height=340)
    mostrar(configurar(grafico, tema))
    linhas = []
    for algoritmo, grupo in dados.groupby("algoritmo"):
        if grupo["fator"].nunique() < 2:
            continue
        ajuste = ajuste_potencia(grupo["fator"], grupo[medida])
        linhas.append(
            {
                "Algoritmo": NOMES[algoritmo],
                "expoente em s": ajuste["expoente"],
                "custo × a cada bit a mais de T": 2 ** ajuste["expoente"],
                "R²": ajuste["r2"],
            }
        )
    if linhas:
        st.dataframe(
            pd.DataFrame(linhas),
            hide_index=True,
            column_config={
                "expoente em s": st.column_config.NumberColumn(format="%.3f"),
                "custo × a cada bit a mais de T": st.column_config.NumberColumn(format="%.2f×"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
            },
        )
    st.caption(
        "Expoente ≈ 1 na DP: custo linear no **valor** de T, logo exponencial no **número de bits** de T (dobra a cada "
        "bit). É isso que se chama pseudo-polinomial, e é por isso que a mochila continua NP-difícil mesmo com uma DP "
        "que, para T de alguns milhares, é rápida."
    )


def aba_qualidade(medidas: pd.DataFrame, algoritmos: list[str], execucao: dict, tema: str) -> None:
    st.markdown(
        "O tempo não é tudo: aqui está **quão perto do ótimo** (soma de scores do Knapsack DP) cada heurística "
        "fica quando a entrada cresce. Só entram seleções que respeitam T (o Top-K padrão às vezes estoura)."
    )
    dados = medidas[(medidas["varredura"] == "principal") & medidas["algoritmo"].isin(algoritmos)]
    validas = dados[~dados["estourou"]]
    heuristicas = [a for a in algoritmos if a != "knapsack"]
    if not heuristicas:
        st.info("O Knapsack DP é o próprio ótimo: escolha ao menos uma heurística na barra lateral para comparar.")
        return
    valores_t, valores_n = sorted(execucao["T"]), sorted(execucao["n"])
    esquerda, direita = st.columns(2, gap="large")
    with esquerda:
        t_fixo = st.select_slider(
            "T fixo", valores_t, value=512 if 512 in valores_t else valores_t[0], key="t_qualidade"
        )
        razao = (
            validas[(validas["T"] == t_fixo) & validas["algoritmo"].isin(heuristicas)]
            .groupby(["algoritmo", "Algoritmo", "n"])["razao_otimo"]
            .mean()
            .reset_index()
        )
        mostrar(
            linhas_simples(
                razao,
                "n",
                "razao_otimo",
                heuristicas,
                tema,
                "n (candidatos)",
                "valor ÷ ótimo (média)",
                ".3f",
                referencias=[(1.0, "ótimo")],
            )
        )
    with direita:
        n_fixo = st.select_slider("n fixo", valores_n, value=valores_n[-1], key="n_qualidade")
        razao = (
            validas[(validas["n"] == n_fixo) & validas["algoritmo"].isin(heuristicas)]
            .groupby(["algoritmo", "Algoritmo", "T"])["razao_otimo"]
            .mean()
            .reset_index()
        )
        mostrar(
            linhas_simples(
                razao,
                "T",
                "razao_otimo",
                heuristicas,
                tema,
                "T (tokens)",
                "valor ÷ ótimo (média)",
                ".3f",
                referencias=[(1.0, "ótimo")],
            )
        )
    st.caption(
        "Greedy: quando T cresce em relação aos pesos, cabem muitos chunks e a perda de no máximo um chunk pesa cada "
        "vez menos na soma, então ele se aproxima do ótimo (garantido pela teoria). Top-K com k fixo faz o contrário: "
        "com mais orçamento e mais candidatos, ele continua levando k chunks e se afasta do ótimo. "
        "Cada ponto é a média "
        "das seleções que respeitam T."
    )
    st.markdown("##### Garantia do Greedy: perda ≤ valor de um chunk")
    perdas = validas[validas["algoritmo"].isin(heuristicas)].assign(
        perda_relativa=lambda d: np.where(d["v_max"] > 0, d["perda"] / d["v_max"], 0.0)
    )
    tinta = TINTA[tema]
    # Os extremos do Top-K (dezenas de chunks de perda) achatariam o Greedy: o eixo vai até 3 chunks.
    teto = 3.0
    caixa = (
        alt.Chart(perdas)
        .mark_boxplot(size=26, extent="min-max", median={"color": tinta["texto"]})
        .encode(
            x=eixo_x_algoritmo(heuristicas),
            y=alt.Y(
                "perda_relativa:Q",
                title="(ótimo − valor) ÷ maior score que cabe",
                scale=alt.Scale(domain=[0, teto], clamp=True),
            ),
            color=cor(heuristicas, tema, legenda=False),
        )
    )
    limite = alt.Chart(pd.DataFrame({"y": [1.0]})).mark_rule(color=tinta["texto"], strokeWidth=1.5).encode(y="y:Q")
    mostrar(configurar(alt.layer(caixa, limite).properties(height=320), tema))
    cumpre = perdas.groupby("Algoritmo")["perda_relativa"].apply(lambda x: float((x <= 1 + 1e-9).mean())).reset_index()
    cumpre.columns = ["Algoritmo", "casos com perda ≤ 1 chunk"]
    extremos = perdas.groupby("Algoritmo")["perda_relativa"].agg(["max", lambda x: x.quantile(0.95)]).reset_index()
    extremos.columns = ["Algoritmo", "perda máxima (chunks)", "percentil 95 (chunks)"]
    cumpre = cumpre.merge(extremos, on="Algoritmo")
    st.dataframe(
        cumpre,
        hide_index=True,
        column_config={
            "casos com perda ≤ 1 chunk": st.column_config.NumberColumn(format="percent"),
            "perda máxima (chunks)": st.column_config.NumberColumn(format="%.2f"),
            "percentil 95 (chunks)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        f"A linha marca o limite teórico do Greedy. Top-K e Top-K com corte não têm essa garantia; se passarem "
        f"do limite, não é erro. O eixo vai até {num(teto, 0)} chunks para o Greedy ficar legível; os valores "
        "maiores do Top-K estão na tabela."
    )
    if "topk" in algoritmos:
        estouro = dados[dados["algoritmo"] == "topk"].groupby(["n", "T"])["estourou"].agg(["mean", "size"])
        estouro = estouro.reset_index().rename(columns={"mean": "mediana", "size": "casos"})
        st.markdown("##### Top-K padrão: em quantos casos estoura T")
        mostrar(
            mapa_de_calor(
                estouro,
                tema,
                "fração de casos que estouram T",
                "estouro",
                lambda v: f"{v:.0%}",
            )
        )


def aba_metodologia(dados: dict, execucao: dict, execucoes: list[Path]) -> None:
    maquina = execucao.get("maquina", {})
    casos_memoria = _grade(execucao, "casos_memoria", "?")
    st.markdown(
        f"""
**Como os casos são montados.** Cada um dos {execucao["casos"]} casos é uma pergunta do HotpotQA ({execucao["fonte"]}).
A instância de tamanho n tem os parágrafos da própria pergunta e, além deles, parágrafos reais de outras perguntas
(com seus pesos em tokens e scores reais), sorteados numa ordem fixa por caso (semente {execucao["semente"]}). As
instâncias de um caso são aninhadas (a de n = 50 contém a de n = 20). Pool: {execucao["pool"]} perguntas.

**Tempo.** `time.perf_counter_ns`, com o coletor de lixo desligado durante a medição (como no `timeit`) e o processo
fixo num núcleo ({maquina.get("afinidade", "não registrado")}; governador de frequência
{maquina.get("governador", "não registrado")}). Uma execução de aquecimento e repetições até somar 2 ms (no máximo 50);
vale a mediana. Seleções de 20 ms ou mais são medidas uma vez. Os gráficos mostram a mediana entre os casos e a faixa
de 25% a 75%.

**Operações.** Cada algoritmo tem uma réplica instrumentada que executa os mesmos laços e as mesmas ordenações do
código em `seletores.py` e conta: comparações feitas pelo Timsort (chaves que contam cada `<`), passos de laço e de
cálculo de chaves, células da DP (o laço interno é de fato executado), melhorias (células que gravam valor) e posições
de memória zeradas na alocação. **Em toda medição o experimento confere que a réplica escolhe exatamente os mesmos
chunks, na mesma ordem, que o código real**; se divergir, a execução para.

**Memória.** Pico de alocação do `tracemalloc` numa seleção, em {casos_memoria} casos por configuração.

**Ajustes.** Mínimos quadrados em escala log: y ≈ c·nᵃ·Tᵇ (varredura principal) e y ≈ c·baseⁿ (força bruta), sobre
as medianas. Os expoentes do Resumo usam só as entradas grandes (ajustável), onde custos fixos pesam menos.

**Máquina.** {maquina.get("processador", "?")} · {maquina.get("nucleos", "?")} núcleos · Python
{maquina.get("python", "?")} · {maquina.get("sistema", "?")}. Execução de {execucao["data"]}, levou
{num(execucao.get("duracao_s", 0) / 60, 1)} min. Embeddings: {execucao["embedder"]}; tokens: {execucao["tokenizador"]}.

**Limitações.** O tempo de relógio depende da máquina e de outros processos (as operações não). O Python tem custo fixo
por chamada que domina em n pequeno. Os expoentes são estimativas numa faixa finita de n e T; os expoentes locais
mostram se a curva já se estabilizou, e a tabela abaixo compara execuções independentes.
"""
    )
    st.markdown("**Reprodutibilidade: expoentes de cada execução (entradas grandes)**")
    linhas = []
    for pasta in execucoes:
        try:
            outra = json.loads((pasta / "execucao.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for ajuste in outra.get("ajustes", []):
            if ajuste.get("regime") == "entradas grandes" and "expoente_n" in ajuste:
                linhas.append(
                    {
                        "execução": pasta.name.removeprefix("assintotica-"),
                        "medida": MEDIDAS.get(ajuste["medida"].replace("tempo_ns", "tempo_us"), ajuste["medida"]),
                        "Algoritmo": NOMES[ajuste["algoritmo"]],
                        "expoente em n": ajuste["expoente_n"],
                        "expoente em T": ajuste["expoente_T"],
                        "R²": ajuste["r2"],
                    }
                )
    if linhas:
        st.dataframe(
            pd.DataFrame(linhas),
            hide_index=True,
            column_config={
                "expoente em n": st.column_config.NumberColumn(format="%.3f"),
                "expoente em T": st.column_config.NumberColumn(format="%.3f"),
                "R²": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("Com mais de uma execução, a diferença entre elas mostra quanto os expoentes de tempo variam.")
    else:
        st.caption("Nenhuma execução registrou ajustes nas entradas grandes.")
    tabela = dados["tempo_us"].merge(
        dados["operacoes"][["varredura", "algoritmo", "n", "T", "mediana"]].rename(columns={"mediana": "operacoes"}),
        on=["varredura", "algoritmo", "n", "T"],
    )
    tabela = tabela[["varredura", "Algoritmo", "n", "T", "mediana", "p25", "p75", "operacoes", "casos"]].rename(
        columns={"mediana": "tempo_mediano_us", "p25": "tempo_p25_us", "p75": "tempo_p75_us"}
    )
    st.markdown("**Medianas por configuração**")
    st.dataframe(tabela, hide_index=True, height=360)
    st.download_button(
        "Baixar como CSV", tabela.to_csv(index=False).encode("utf-8"), "assintotica_medianas.csv", "text/csv"
    )


# ---------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Análise assintótica", page_icon="📈", layout="wide")
    st.title("Análise assintótica dos seletores de contexto")
    tema = tema_atual()
    execucoes = listar_execucoes(PASTA_RESULTADOS)
    if not execucoes:
        st.warning(f"Nenhuma análise em {PASTA_RESULTADOS}. Rode `uv run task assintotica` para gerar os dados.")
        aba_teoria()
        return
    with st.sidebar:
        pasta = st.selectbox("Execução", execucoes, format_func=lambda p: p.name.removeprefix("assintotica-"))
    dados = carregar(str(pasta), (pasta / "medidas.jsonl").stat().st_mtime)
    execucao = dados["execucao"]
    with st.sidebar:
        presentes = [a for a in PRINCIPAIS if a in set(dados["medidas"]["algoritmo"])]
        algoritmos = st.multiselect("Algoritmos", presentes, default=presentes, format_func=NOMES.get)
        st.divider()
        st.caption(
            f"{execucao['casos']} casos · n ∈ {execucao['n']} · T ∈ {execucao['T']} · k = {execucao['k']}  \n"
            f"Força bruta: n até {max(config_forca_bruta(execucao)['n'] or [0])} "
            f"({config_forca_bruta(execucao)['casos']} casos)  \n"
            f"{len(dados['medidas']):,} medidas".replace(",", ".")
        )
    if not algoritmos:
        st.info("Escolha ao menos um algoritmo na barra lateral.")
        return
    abas = st.tabs(
        [
            "Resumo",
            "Teoria",
            "Tempo × n",
            "Tempo × T",
            "Operações",
            "Mapa n × T",
            "Força bruta × DP",
            "Melhor × pior caso",
            "Pseudo-polinomial",
            "Memória",
            "Qualidade × tamanho",
            "Metodologia e dados",
        ]
    )
    with abas[0]:
        aba_resumo(dados, algoritmos, execucao, tema)
    with abas[1]:
        aba_teoria()
    with abas[2]:
        aba_crescimento(dados, algoritmos, execucao, tema, "n")
    with abas[3]:
        aba_crescimento(dados, algoritmos, execucao, tema, "T")
    with abas[4]:
        aba_operacoes(dados, algoritmos, execucao, tema, dados["medidas"])
    with abas[5]:
        aba_mapa(dados, algoritmos, tema)
    with abas[6]:
        aba_forca_bruta(dados, execucao, tema)
    with abas[7]:
        aba_cenarios(dados, algoritmos, execucao, tema)
    with abas[8]:
        aba_escala(dados, algoritmos, execucao, tema)
    with abas[9]:
        aba_memoria(dados, algoritmos, execucao, tema)
    with abas[10]:
        aba_qualidade(dados["medidas"], algoritmos, execucao, tema)
    with abas[11]:
        aba_metodologia(dados, execucao, execucoes)


main()
